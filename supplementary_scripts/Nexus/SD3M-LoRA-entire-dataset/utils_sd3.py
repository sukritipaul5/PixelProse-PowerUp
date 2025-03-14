
import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import ipdb
import time

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from accelerate.state import DistributedType
from torch.utils.data import DistributedSampler
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image,ImageOps
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

#import zstandard as zstd
from PIL import Image
import io
import prodigyopt
import deepspeed
import safetensors.torch




def log_memory_usage():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def check_disk_space(path):
    total, used, free = shutil.disk_usage(path)
    print(f"Total: {total // (2**30)} GB")
    print(f"Used: {used // (2**30)} GB")
    print(f"Free: {free // (2**30)} GB")

def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch



def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def unwrap_model(model,accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model



# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir,accelerator,transformer,text_encoder_one,text_encoder_two):
    if accelerator.is_main_process:
        transformer_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None

        for model in models:
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_lora_layers_to_save = get_peft_model_state_dict(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        StableDiffusion3Pipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )



def load_model_hook(models, input_dir,transformer,text_encoder_one,text_encoder_two,args):
    transformer_ = None
    text_encoder_one_ = None
    text_encoder_two_ = None

    while len(models) > 0:
        model = models.pop()

        if isinstance(model, type(unwrap_model(transformer))):
            transformer_ = model
        elif isinstance(model, type(unwrap_model(text_encoder_one))):
            text_encoder_one_ = model
        elif isinstance(model, type(unwrap_model(text_encoder_two))):
            text_encoder_two_ = model
        else:
            raise ValueError(f"unexpected save model: {model.__class__}")

    lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    if args.train_text_encoder:
        # Do we need to call `scale_lora_layers()` here?
        _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        _set_state_dict_into_text_encoder(
            lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
        )

    # Make sure the trainable params are in float32. This is again needed since the base models
    # are in `weight_dtype`. More details:
    # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
    if args.mixed_precision == "fp16":
        models = [transformer_]
        if args.train_text_encoder:
            models.extend([text_encoder_one_, text_encoder_two_])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models)


def get_sigmas(accelerator,noise_scheduler_copy,timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma



def save_lora_weights(save_path,global_step,transformer,accelerator):
    print(f"Starting save for step {global_step}")
    
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    print(f"Created directory: {save_path}")                        
    # Get the unwrapped model
    unwrapped_model = accelerator.unwrap_model(transformer)                        
    # Save the LoRA weights
    state_dict = unwrapped_model.state_dict()
    safetensors.torch.save_file(state_dict, os.path.join(save_path, "pytorch_lora_weights.safetensors"))                        
    print(f"Saved LoRA weights to {os.path.join(save_path, 'pytorch_lora_weights.safetensors')}")
    print(f"Completed save for step {global_step}")     


def load_text_encoders(args,class_one, class_two, class_three):
    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three
