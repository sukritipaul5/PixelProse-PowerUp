
# PixelProse PowerUp: SD3-Medium Full Fine-tuning and Context Window Extension
We're fine-tuning Stable Diffusion 3 Medium  on our PixelProse dataset with the research goal of enhancing text rendering capabilities in large diffusion models. This project extends the text context window length of SD3-M, allowing for longer and more detailed text prompts in image generation. 
Specifically, we're aiming for high GenEval scores across categories pertaining to text rendering, counts, shape, color, and object relations. 
<img width="975" alt="image" src="https://github.com/user-attachments/assets/27e8ee7a-8e8b-4915-b6f2-9071567e21aa">

## 🚀 Features
- Full fine-tuning with custom dataset & dataloader
- Extended context window for SD3-Medium
- Optimizations for memory & runtime (caching text latents for faster I/O, distributed data sharding, and offloading model weights to CPU)
- Validation & checkpointing 
- Gradient accumulation and mixed precision training
- Cosine warm-up schedule for trainable parameters


## 🛠️ Tech Specs

- Model: Stable Diffusion 3 Medium
- GPU: NVIDIA A100s 
- Cluster: Polaris & Nexus
- Framework: PyTorch with DeepSpeed
- Dataset: PixelProse 16M with cached latents

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/sukritipaul5/PixelProse-PowerUp.git
cd PixelProse-PowerUp/scripts/SD3-Medium-ContextWindow
```

Install dependencies:

```bash
Will add soon
```


## Fine-tuning SD3-M
You can visit the [scripts/](https://github.com/sukritipaul5/PixelProse-PowerUp/tree/main/scripts/) directory to find the scripts we used to fine-tune SD3-M on our dataset. The  two folders correpond to full fine-tune and context window extension respectively.
You can ```qsub``` the job scripts as follows:

```bash
qsub SD3-Medium-ContextWindow/launch.sh
qsub SD3-Medium-FFT/launch.sh
```
#### Change the arguments in the launch script as needed 
```python
python train_sd3m_cw.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-base" \
    --output_dir="./sd3m_context_window" \
    --train_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=1e-4 \
    --max_grad_norm=1.0 \
    --validation_step=500 \
    --validation_prompts_file="/path/to/validation_prompts.txt" \
    --mixed_precision="bf16" \
    --gradient_accumulation_steps=4 \
    --seed=42
```

### Scripts
- `train_sd3m_cw.py`: Main training script
- `utils_sd3.py`: Utility functions for SD3
- `datapipe_new.py`: Custom data pipeline for PixelProse dataset with cached latents

### 🧠 How it works
- Extended Context Window: We use a custom MLP to process extended context beyond the standard 77 tokens.
- Custom Dataloader: Our `WebDatasetwithCachedLatents` class efficiently handles large-scale data with pre-computed latents.
- Training Loop: The main training loop in `train_sd3m_cw.py` handles gradient accumulation, mixed precision training, and periodic validation.
- Validation: During training, we generate images using the current model state to track progress visually on wandb.


### 🎛️ Parameters
Key parameters in  `train_sd3m_cw.py`:


- `--pretrained_model_name_or_path`: Path to the pretrained SD3-M model
- `--output_dir`: Directory to save checkpoints and logs
- `--train_batch_size`: Batch size for training
- `--learning_rate`: Initial learning rate
- `--max_grad_norm`: Maximum gradient norm for clipping
- `--validation_step`: Number of steps between validations
- `--mixed_precision`: Precision for mixed precision training (fp16, bf16, or no)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating model weights
- `--seed`: Random seed for reproducibility
- `--validation_prompts_file`: Path to the file containing validation prompts
- `--num_train_epochs`: Number of training epochs
- `--cosine_warmup_steps`: Number of steps for cosine warm-up schedule
- `--epsilon`: Small constant for numerical stability in optimizer
- `--weight_decay`: Constant for L2 regularization
- `--lr_scheduler_type`: Type of learning rate scheduler (cosine, linear, etc.)
- `--checkpoint_dir`: Directory to save checkpoints
- `--checkpoint_steps`: Number of steps between checkpoints


### 🤝 Contributions are welcome! Please feel free to submit a PR!


## References & Credits
- Stability AI for the original SD3-M model.
- The Hugging Face team for their Diffusers library and their dreambooth finetuning repo.
- Databricks for MosaicML and efficient data sharding strategies.
- ALCF for compute.

copyright (c) 2024 TomLab
