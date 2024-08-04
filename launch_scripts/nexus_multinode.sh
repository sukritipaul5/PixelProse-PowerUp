#!/bin/bash
#SBATCH --job-name=accelerate_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --output=/fs/nexus-scratch/sukriti5/nexus_multinode/log.out
#SBATCH --error=/fs/nexus-scratch/sukriti5/nexus_multinode/log.err



##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "



# ******************* Don't change this **************************************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work
# Find an available port
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR


# Debug 
export NCCL_DEBUG=INFO
export ACCELERATE_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT_SEC=300
export TORCH_DISTRIBUTED_TIMEOUT=300
export NCCL_NET_GDR_LEVEL=2


# Use TCP instead of shared memory
#Must disable NCCL_IB
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

#Super important to export CUDA visible devices. Otherwise, only multi-node works and not multi-gpu.
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "Run begins:$(date)"

srun --cpu_bind=none --kill-on-bad-exit=0 accelerate launch \
    --multi_gpu \
    --num_processes $WORLD_SIZE \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend c10d \
    simple_model.py


echo "Run ends: $(date)"
