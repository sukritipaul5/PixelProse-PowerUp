# Nexus single-node setup for SD3M-LoRA Fine-tune 
### Single Node (Test Run)

1. Main train script: `temp_sd3.py`
2. Accelerate config: `deep_speed_sd3.yaml`
3. Environment path: `/fs/cml-projects/yet-another-diffusion/sd3m-diffusion/env/sd3-medium-lora`


### Interative Command
Interactive run: `srun --pty --gres=gpu:rtxa6000:1 --cpus-per-task=4 --mem=128G --qos=scavenger --account=scavenger --partition=scavenger --time=1:00:00 /bin/bash`


### Launch Specs (for Interactive)
```
module load cuda/12.1.1
conda activate /fs/cml-projects/yet-another-diffusion/sd3m-diffusion/env/sd3-medium-lora

export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export OUTPUT_DIR="/fs/cml-projects/yet-another-diffusion/sd3m-diffusion/outputs/run04-sd3-lora"

export HF_TOKEN="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"
export WANDB_API_KEY="fdd9c522e7590584e032c3be997620b217e44bd0"
export WANDB_MODE=ONLINE
export ACCELERATE_CONFIG_FILE='/fs/cml-projects/yet-another-diffusion/sd3m-diffusion/accelerate_configs/deep_speed_sd3.yaml'

huggingface-cli login --token $HF_TOKEN

cd /fs/cml-projects/yet-another-diffusion/diffusers/examples/dreambooth/

accelerate launch temp_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name="lambdalabs/naruto-blip-captions" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=4 \
  --sample_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=0.0001 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=50 \
  --checkpointing_steps=5 \
  --gradient_checkpointing \
  --weighting_scheme="logit_normal" \
  --seed=42 \
  --prior_generation_precision="fp16" \
  --use_custom_prompts=True \
  --dataloader_num_workers=0 \
  --adam_weight_decay=1e-02 \
  --max_sequence_length=77 \
  --max_train_steps=100 \
  --num_samples=5000
```

### Data subset used for Test Run
1. Test subset: 5k samples from `/fs/cml-projects/yet-another-diffusion/pixelprose-shards/commonpool_node0_part1/`
2. Throughput : 3.49
