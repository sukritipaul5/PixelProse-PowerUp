#!/usr/bin/env python
# coding=utf-8
# SD3-M Dreambooth: Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# This script has been modified to support fine-tuning SD3-M with LoRA (Low-Rank Adaptation) 
# by Tom's Lab.
#
# Modifications and contributions by:
# Sukriti Paul, Vasu Singla
# 
# For more information or inquiries, please contact the authors.

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
import datetime
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
import torch.distributed as dist

from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
import webdataset as wds
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
import os
import utils_sd3
from data_sd3_embed import StatefulWebDataset, create_dataloader
import wandb
import psutil
#from memory_profiler import profile



# if is_wandb_available():
#     import wandb

# Will error if the minimal version of diffusers is not installed. 
check_min_version("0.30.0.dev0")


logger = get_logger(__name__)
os.environ['WANDB_MODE'] = 'online'
os.environ['LD_LIBRARY_PATH'] = '/soft/compilers/cudatoolkit/cuda-12.2.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/soft/compilers/cudatoolkit/cuda-12.2.2'



def log_memory_usage(step):
    process = psutil.Process(os.getpid())
    logging.info(f"Step {step}: CPU Memory: {process.memory_info().rss / 1e9:.2f} GB")


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="A cat driving a car, high detail, 8K resolution.",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )

    parser.add_argument(
        "--validation_step",
        type=int,
        default=500,
        help=("Use iters instead" ),
    )


    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder (clip text encoders only). If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--use_custom_prompts",
        type=bool,
        default=True,
        help="Use if each image has its own prompts.",
    )

    parser.add_argument(
    "--num_samples",
    type=int,
    default=1000,
    help="Number of samples to use from the dataset",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # logger is not available yet
    if args.class_data_dir is not None:
        warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
    if args.class_prompt is not None:
        warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args



def main(args):


    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(
            os.path.join(args.output_dir, "log.txt"), mode="a"
        )]
    )
    logging.info("Starting main function")
    logging.info("Initializing accelerator")


    if dist.is_available() and dist.is_initialized():
        # In distributed setting, only report on rank 0
        if dist.get_rank() == 0:
            args.report_to = "wandb"
        else:
            args.report_to = None
    else:
        # In non-distributed setting (including single GPU), always report
        args.report_to = "wandb"


    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, args.logging_dir)
    )


    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )


    #Wandb online logging
    if accelerator.is_main_process:
        import wandb
        wandb.init(project="sd3medium-lora-finetune", config=args)

    logger.info(f"Distributed environment: {accelerator.distributed_type}")
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        logger.info("DeepSpeed is enabled")
    else:
        logger.info("DeepSpeed is not enabled")


    logger.info(f"Distributed environment: {accelerator.distributed_type}")
    logger.info(f"Num processes: {accelerator.num_processes}")
    logger.info(f"Process index: {accelerator.process_index}")
    logger.info(f"Local process index: {accelerator.local_process_index}")
    logger.info(f"Device: {accelerator.device}")

    
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )


    logger.info(f"Distributed environment: {accelerator.distributed_type}")
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        logger.info("DeepSpeed is enabled")
    else:
        logger.info("DeepSpeed is not enabled")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Handle the repository creation
    if accelerator.is_main_process:
        #wandb.log({"loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)        
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    logging.info("Loading tokenizers")
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        #torch_dtype = torch.bfloat16,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        #torch_dtype = torch.bfloat16,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
        #torch_dtype = torch.bfloat16,    
    )   

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = utils_sd3.load_text_encoders(args,
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        #torch_dtype=torch.bfloat16
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    # Register the save hook
    accelerator.register_save_state_pre_hook(
        lambda models, weights, output_dir: utils_sd3.save_model_hook(
            models, weights, output_dir, accelerator, transformer, text_encoder_one, text_encoder_two
        )
    )

    # Register the load hook
    accelerator.register_load_state_pre_hook(
        lambda models, input_dir: utils_sd3.load_model_hook(
            models, input_dir, accelerator, transformer, text_encoder_one, text_encoder_two
        )
    )
        

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"
 

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
   
    
    # Dataset and DataLoaders creation:
    train_dataset = StatefulWebDataset(
        # tar_path="/fs/cml-projects/yet-another-diffusion/pixelprose-shards/redcaps_part*/*.tar",
        # latents_dir="/fs/cml-projects/yet-another-diffusion/pixelprose_sd3_latents/",
        tar_path="/fs/cml-projects/yet-another-diffusion/pixelprose-shards/*/*.tar",
        latents_dir="/fs/cml-projects/yet-another-diffusion/pixelprose_sd3_latents/",
        tokenizers=tokenizers,
        size=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        max_length=args.max_sequence_length,
        num_samples=args.num_samples
    )

    logging.info(f"Number of samples to process: {args.num_samples}")
    logging.info("Preparing data loader")

    #logging.info(DistributedType.NO)
    log_memory_usage("Before dataloader creation")
    train_dataloader, estimated_num_batches = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        distributed=accelerator.distributed_type != DistributedType.NO
    )
    logger.info(f"Process {accelerator.process_index} dataloader initialized")
    log_memory_usage("Before dataloader creation")


    log_memory_usage("Before first batch fetch")
    try:
        first_batch = next(iter(train_dataloader))
        log_memory_usage("After first batch fetch")
        print("First batch keys:", first_batch.keys())
        for k, v in first_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k} shape: {v.shape}, dtype: {v.dtype}")
    except Exception as e:
        print(f"Error fetching first batch: {str(e)}")

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]


    # For MDS dataset, we don't pre-compute embeddings because each image has a unique prompt
    logger.info("Using custom prompts for each image.")
    prompt_embeds = None
    pooled_prompt_embeds = None
    

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(args.num_samples / (args.train_batch_size * args.gradient_accumulation_steps))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    args.max_train_steps = min(args.max_train_steps, num_update_steps_per_epoch * args.num_train_epochs)


    lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes * args.gradient_accumulation_steps,
    num_training_steps=args.max_train_steps * accelerator.num_processes * args.gradient_accumulation_steps,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
    )


    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
        assert text_encoder_one is not None
        assert text_encoder_two is not None
        assert text_encoder_three is not None
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(args.num_samples / (args.train_batch_size * args.gradient_accumulation_steps))

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # We cap the number of training steps to ensure we don't exceed the number of available samples
   
    args.max_train_steps = min(args.max_train_steps, num_update_steps_per_epoch * args.num_train_epochs)

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
 
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sd3-lora-finetune"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
  

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    #logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    if estimated_num_batches is not None:
        logger.info(f"  Estimated num batches each epoch = {estimated_num_batches}")
    else:
        logger.info("  Num batches each epoch = Unknown (infinite dataset)")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

            # Load dataset state
            dataset_state = torch.load(os.path.join(args.output_dir, path, "dataset_state.pt"))
            if hasattr(train_dataset, 'load_state_dict'):
                train_dataset.load_state_dict(dataset_state)

            # Recreate the dataloader
            train_dataloader = accelerator.prepare(
                create_dataloader(
                    train_dataset,
                    batch_size=args.train_batch_size,
                    num_workers=args.dataloader_num_workers,
                    distributed=accelerator.distributed_type != DistributedType.NO
                )
            )         
            
    else:
        initial_global_step = 0
        first_epoch = 0
        resume_step = 0
  


    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    
    # Number of arrays in the dataloader
    if estimated_num_batches is not None:
        
        num_arrays = estimated_num_batches
    else:
        num_arrays = float('inf') 


    # Shape of each array
    array_shape = (args.train_batch_size, 154, 4096)
   

    prompt_embeds_list = []
    # Check if there is a remainder
    remainder = len(train_dataset) % args.train_batch_size
   

    # Fill the list with tensors of the array_shape
    for i in range(num_arrays):
        if i == num_arrays - 1 and remainder != 0:
            # Create a tensor for the final batch with the remainder size
            final_batch_shape = (remainder, 154, 4096)
            final_batch_tensor = torch.empty(final_batch_shape, dtype=torch.bfloat16).to("cpu")
            prompt_embeds_list.append(final_batch_tensor)
        else:
            tensor = torch.empty(array_shape, dtype=torch.bfloat16).to("cpu")
            prompt_embeds_list.append(tensor)

   

    global_start_time = time.time()
    last_logging_time = global_start_time
    total_samples = 0
    samples_since_last_log = 0



    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"Starting epoch {epoch}")
        #temp added
        # if accelerator.distributed_type != DistributedType.NO:
        #     logger.info("Waiting for barrier...")
        #     dist.barrier()
        #     logger.info("Passed barrier")

        #temp added
        #if distributed:
        #train_dataloader.sampler.set_epoch(epoch)

        transformer.train()
        logger.info("Set transformer to train mode")


        logger.info("About to start iterating over dataloader")   
        logger.info(f"Estimated number of batches: {estimated_num_batches}")


        for step, batch in enumerate(train_dataloader):
            #logger.info(f"Got batch {step}")
            if batch is None:
                #print("Encountered a None batch, skipping...")
                continue
            
            #debug temporary
            #logger.info(f"Rank {dist.get_rank() if dist.is_initialized() else 0} - Epoch {epoch}, Step {step}")
            
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=accelerator.device)

                prompt_embeds = batch["prompt_embeds"].to(device=accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=accelerator.device)
                #print(prompt_embeds.shape,pooled_prompt_embeds.shape)

                #debug temporary- note this is not taking in more than 1 batch irrespective of train_batch_size.
                # logger.info(f"Batch keys: {batch.keys()}")
                # logger.info(f"Pixel values shape: {batch['pixel_values'].shape}")
                # logger.info(f"Prompt embeds shape: {batch['prompt_embeds'].shape}")
                # logger.info(f"Pooled prompt embeds shape: {batch['pooled_prompt_embeds'].shape}")

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                
                # Add noise according to flow matching.
                sigmas = utils_sd3.get_sigmas(accelerator,noise_scheduler_copy,timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
           
                

                if len(model_input) != len(prompt_embeds):
                    prompt_embeds = prompt_embeds[:len(model_input),:,:]

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                # del prompt_embeds,pooled_prompt_embeds
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                # flow matching loss
                target = model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()                

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                current_time = time.time()
                time_since_last_log = current_time - last_logging_time

                #added
                batch_size = len(batch["pixel_values"])
                samples_this_step = batch_size * accelerator.num_processes
                samples_since_last_log += samples_this_step
                total_samples += samples_this_step

                throughput = samples_since_last_log / time_since_last_log
                
                # Gather metrics from all processes
                loss_tensor = torch.tensor([loss.detach().item()], device=accelerator.device)
                throughput_tensor = torch.tensor([throughput], device=accelerator.device)

                if accelerator.distributed_type != DistributedType.NO:
                    dist.barrier()
                    dist.all_reduce(loss_tensor)
                    dist.all_reduce(throughput_tensor)
                    
                    # Average the loss
                    world_size = accelerator.num_processes
                    loss_tensor /= world_size
                    throughput_tensor /= world_size

     
                # Logging
                if accelerator.is_main_process:
                    logs = {
                        "loss": loss_tensor.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "throughput": throughput_tensor.item() ,
                        "total_samples": total_samples,
                        "elapsed_time": current_time - global_start_time,
                    }
                    wandb.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                
                last_logging_time = current_time
                samples_since_last_log = 0

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        # Save dataset state instead of dataloader state
                        dataset_state = train_dataset.state_dict() if hasattr(train_dataset, 'state_dict') else {}
                        torch.save(dataset_state, os.path.join(save_path, "dataset_state.pt"))

                # Validation
                if accelerator.is_main_process:
                    condition = (args.validation_prompt is not None and (global_step % args.validation_step==0))
                    if condition:
                        print(global_step,args.validation_step)
                        logger.info(f"Running validation... \nEpoch: {epoch}, Global Step: {global_step}")
                        logger.info(f"Starting validation at step {global_step}")

                        text_encoders = text_encoder_one,text_encoder_two,text_encoder_three
                        utils_sd3.validate_log(args,accelerator,vae,text_encoders,\
                            transformer,global_step,weight_dtype,logger=logger)

            if global_step >= args.max_train_steps:
                break          
    
        # Check for early stopping
        if global_step >= args.max_train_steps or global_step >= args.num_samples // args.train_batch_size:
            break
    
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = utils_sd3.unwrap_model(transformer,accelerator)
        transformer = transformer.to(torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir, transformer_lora_layers=transformer_lora_layers
        )

        utils_sd3.validate_log(args,accelerator,vae,text_encoders,\
            transformer,global_step,weight_dtype,logger=logger)


        """
            Sanity Check: Loads the saved model and runs inference.
        """

        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # gc.collect()

        # pipeline = StableDiffusion3Pipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     revision=args.revision,
        #     variant=args.variant,
        #     torch_dtype=weight_dtype,
        # )
        # # load attention processors
        # pipeline.load_lora_weights(args.output_dir)

        # # run inference
        # #logger.info("Temporarily disabling validation...")
        # logger.info("Attempt at validation")
        # # images = []
        # if args.validation_prompt and args.num_validation_images > 0:
        #     pipeline_args = {"prompt": args.validation_prompt}
        #     images = utils_sd3.log_validation(
        #         pipeline=pipeline,
        #         args=args,
        #         accelerator=accelerator,
        #         global_step=global_step,
        #         logger=logger,
        #         is_final_validation=True,
        #     )




    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

