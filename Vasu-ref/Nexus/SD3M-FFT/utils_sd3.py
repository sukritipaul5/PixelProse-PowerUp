
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

from PIL import Image
import io
import prodigyopt
import deepspeed
import safetensors.torch
import wandb



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

    
    print(prompt_embeds.shape,pooled_prompt_embeds.shape)
    breakpoint()
    
    return prompt_embeds, pooled_prompt_embeds


def unwrap_model(model,accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model



# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a decent format
# def save_model_hook(models, weights, output_dir, accelerator, transformer, text_encoder_one, text_encoder_two):
#     if accelerator.is_main_process:
#         transformer_lora_layers_to_save = None
#         text_encoder_one_lora_layers_to_save = None
#         text_encoder_two_lora_layers_to_save = None

#         for model in models:
#             unwrapped_model = unwrap_model(model, accelerator)
#             if isinstance(unwrapped_model, type(unwrap_model(transformer, accelerator))):
#                 transformer_lora_layers_to_save = get_peft_model_state_dict(unwrapped_model)
#             elif isinstance(unwrapped_model, type(unwrap_model(text_encoder_one, accelerator))):
#                 text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(unwrapped_model)
#             elif isinstance(unwrapped_model, type(unwrap_model(text_encoder_two, accelerator))):
#                 text_encoder_two_lora_layers_to_save = get_peft_model_state_dict(unwrapped_model)
#             else:
#                 raise ValueError(f"unexpected save model: {unwrapped_model.__class__}")

#             #pop weight so that corresponding model is not saved again!
#             weights.pop()

#         StableDiffusion3Pipeline.save_lora_weights(
#             output_dir,
#             transformer_lora_layers=transformer_lora_layers_to_save,
#             text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
#             text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
#         )


# def load_model_hook(models, input_dir, accelerator, transformer, text_encoder_one, text_encoder_two):
#     lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

#     for model in models:
#         if isinstance(model, type(unwrap_model(transformer))):
#             transformer_state_dict = {
#                 f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
#             }
#             transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
#             set_peft_model_state_dict(model, transformer_state_dict, adapter_name="default")
#         elif isinstance(model, type(unwrap_model(text_encoder_one))):
#             _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=model)
#         elif isinstance(model, type(unwrap_model(text_encoder_two))):
#             _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder_2.", text_encoder=model)
#         else:
#             raise ValueError(f"unexpected load model: {model.__class__}")

#     if accelerator.mixed_precision == "fp16":
#         cast_training_params([transformer, text_encoder_one, text_encoder_two])



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


def validate_log(args, accelerator, vae, text_encoders, transformer, global_step, weight_dtype, logger):
    text_encoder_one, text_encoder_two, text_encoder_three = text_encoders

    torch.cuda.empty_cache()
    
    # Create the pipeline with the fine-tuned transformer
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,  # No need to unwrap as these are not trained
        text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three,
        transformer=transformer,  # This is already unwrapped in the main script
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    
    logger.info("Pipeline created for validation")
    
    # Move the pipeline to the appropriate device
    pipeline = pipeline.to(accelerator.device)
    
    # Set the pipeline to evaluation mode
    pipeline.set_progress_bar_config(disable=True)
    
    # Generate validation images
    with torch.no_grad():
        images = pipeline(
            prompt=args.validation_prompt,
            num_inference_steps=30,
            num_images_per_prompt=args.num_validation_images,
            generator=torch.Generator(device=accelerator.device).manual_seed(args.seed)
        ).images
    
    # Log the validation images (assuming you have a log_validation function)
    log_validation(
        pipeline=pipeline,
        args=args,
        accelerator=accelerator,
        global_step=global_step,
        logger=logger,
        images=images
    )
    
    logger.info("Validation completed")
    
    # Clean up
    del pipeline
    torch.cuda.empty_cache()
    logger.info("Cleaned up after validation")

    return images

    
# def validate_log(args,accelerator,vae,text_encoders,transformer,global_step,weight_dtype,logger):

#     text_encoder_one,text_encoder_two,text_encoder_three = text_encoders

#     torch.cuda.empty_cache()
#     pipeline = StableDiffusion3Pipeline.from_pretrained(
#         args.pretrained_model_name_or_path,
#         vae=vae,
#         text_encoder=accelerator.unwrap_model(text_encoder_one),
#         text_encoder_2=accelerator.unwrap_model(text_encoder_two),
#         text_encoder_3=accelerator.unwrap_model(text_encoder_three),
#         transformer=accelerator.unwrap_model(transformer),
#         revision=args.revision,
#         variant=args.variant,
#         torch_dtype=weight_dtype,
#     ) 
#     print("Pipeline created")                   
#     logger.info("Pipeline created")
    
#     pipeline_args = {"prompt": args.validation_prompt}
#     images = log_validation(
#         pipeline=pipeline,
#         args=args,
#         accelerator=accelerator,
#         global_step=global_step,
#         logger=logger
#     )                
#     logger.info("Validation completed")
#     del pipeline
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     logger.info("Cleaned up after validation")


def log_validation(
    pipeline,
    args,
    accelerator,
    global_step,
    logger,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    
    # Move entire pipeline to GPU
    pipeline = pipeline.to(accelerator.device)
    
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    
    images = []
    with torch.no_grad():
        for _ in range(args.num_validation_images):
            image = pipeline(
                prompt=args.validation_prompt, 
                num_inference_steps=30, 
                generator=generator
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":            
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") 
                        for i, image in enumerate(images)
                    ]
                }
            )

    torch.cuda.empty_cache()

    return images