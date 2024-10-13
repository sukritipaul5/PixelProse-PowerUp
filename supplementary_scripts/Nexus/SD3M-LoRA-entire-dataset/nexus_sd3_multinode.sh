#!/usr/bin/env bash
#SBATCH --tasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --job-name=sd3-multinode
#SBATCH --output=sd3-multinode-%j.out
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive

rm -r ./tests

NODES_ARRAY=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')

echo HEAD_NODE_IP = $HEAD_NODE_IP

# envs
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=1 # Nexus does not have Infiniband
export NCCL_SOCKET_IFNAME=bond0
export PYTHONIOENCODING=UTF-8

# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

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

export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="/fs/cml-projects/yet-another-diffusion/diffusers/examples/dreambooth/sd3m-lora-scripts/dog"
export OUTPUT_DIR="./tests"
export SCRIPT_DIR="/fs/cml-projects/yet-another-diffusion/diffusers/examples/dreambooth/sd3m-lora-scripts/temp_sd3.py"

# export ACCELERATE_CONFIG_FILE="accelerate_multinode_config.yaml"

# 2 nodes, 4 gpus per node
# NUM_NODES=2
# NUM_GPUS_PER_NODE=4
# NUM_PROCESSES=$((NUM_NODES * NUM_GPUS_PER_NODE))

# load modules
# source activate /fs/cml-projects/yet-another-diffusion/sd3m-diffusion/env/sd3-medium-lora

module load cuda/12.1.1

echo "SLURM_PROCID = $SLURM_PROCID"

# export ACCELERATE_LAUNCH="accelerate launch \
#     --num_processes 8 \
#     --num_machines 2 \
#     --rdzv_backend c10d \
#     --main-process-ip $HEAD_NODE_IP \
#     --main_process_port 29500 \
#     --machine_rank $SLURM_PROCID \
#     "
# export ACCELERATE_LAUNCH="accelerate launch \
#     --config-file accelerate_multinode_config.yaml \
#     --main-process-ip $HEAD_NODE_IP \
#     --main-process-port 29500 \
#     --machine-rank $SLURM_PROCID \
#     "

WORLD_SIZE=$SLURM_JOB_NUM_NODES

set -x
export LAUNCH="
    torchrun 
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv-endpoint $HEAD_NODE_IP:12384 \
    --nnode $WORLD_SIZE \
    --nproc_per_node 4 \
    "
export SCRIPT="temp_sd3.py"
export SCRIPT_ARGS=" \
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
  "

# launch job
export CMD="$LAUNCH $SCRIPT $SCRIPT_ARGS"
srun $CMD

# srun accelerate launch \
#     --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
#     --num_machines $SLURM_NNODES \
#     --rdzv_backend c10d \
#     --main-process-ip $HEAD_NODE_IP \
#     --main_process_port 29500 \