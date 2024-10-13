#srun --pty -n 1 -c 8  --mem=128gb --gres=gpu:rtxa6000:1 --time=01:00:00 --qos=high   zsh
rm -r /fs/cml-projects/yet-another-diffusion/sd3m-cw-expts/outputs/tests_fft

# NODES_ARRAY=($(scontrol show hostnames $SLURM_JOB_NODELIST))
# HEAD_NODE=${NODES_ARRAY[0]}
# HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')
# echo HEAD_NODE_IP = $HEAD_NODE_IP

# envs
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=1 # Nexus does not have Infiniband
export NCCL_SOCKET_IFNAME=bond0
export PYTHONIOENCODING=UTF-8


# debug
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

echo $pwd
NGPUS=$(nvidia-smi -L | wc -l)
if [ $NGPUS -eq 0 ]; then
    echo "No GPU found"
    exit 1
fi

#export dirs
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export OUTPUT_DIR="/fs/cml-projects/yet-another-diffusion/sd3m-cw-expts/outputs/tests_fft"
#Logging, validation
export HF_TOKEN="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"
export WANDB_API_KEY="fdd9c522e7590584e032c3be997620b217e44bd0"


# load modules
#source activate /fs/cml-projects/yet-another-diffusion/sd3m-diffusion/env/sd3-medium-lora

module unload cuda/12.4.1
module load cuda/12.1.1

echo "SLURM_PROCID = $SLURM_PROCID"
WORLD_SIZE=$SLURM_JOB_NUM_NODES

#HF CLI Login
huggingface-cli login --token $HF_TOKEN


set -x
#export SD3_VALIDATION_PROMPT="cute dog in a bucket"

# export LAUNCH="
#     torchrun 
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv-endpoint $HEAD_NODE_IP:12384 \
#     --nnode $WORLD_SIZE \
#     --nproc_per_node 4 \
#     "

export LAUNCH="python "

export SCRIPT="/fs/cml-projects/yet-another-diffusion/sd3m-cw-expts/scripts-cw/train_sd3m_cw.py"
export SCRIPT_ARGS=" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=100 \
  --checkpointing_steps=200 \
  --gradient_checkpointing \
  --weighting_scheme="logit_normal" \
  --seed=42 \
  --prior_generation_precision="fp16" \
  --use_custom_prompts=True \
  --dataloader_num_workers=0 \
  --adam_weight_decay=1e-02 \
  --adam_epsilon=1e-15 \
  --max_sequence_length=77 \
  --max_train_steps=1000000 \
  --num_samples=100000 \
  --num_validation_images=2 \
  --validation_step=200
  "

# --validation_prompt="cat" \
# launch job
export CMD="$LAUNCH $SCRIPT $SCRIPT_ARGS"
$CMD




