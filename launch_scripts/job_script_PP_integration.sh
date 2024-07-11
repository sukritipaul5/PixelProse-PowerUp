#!/bin/bash
#PBS -N train_sdxl
#PBS -l filesystems=home:eagle
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -A DemocAI
#PBS -o /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_pp_outv2.log
#PBS -e /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_pp_errv2.log

NODES_ARRAY=($(cat "${PBS_NODEFILE}" | sort | uniq))
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(getent hosts $HEAD_NODE | awk 'NR==1 { print $1 }')
HOST_LIST=$(IFS=,; echo "${NODES_ARRAY[*]}")
NNODES=$(wc -l < $PBS_NODEFILE)
NGPUS_PER_NODE=4
NGPUS=$((NGPUS_PER_NODE * NNODES))
NCPUS_PER_GPU=16

export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=12727  # From Kai's code
export HF_TOKEN="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"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export WANDB_API_KEY="fdd9c522e7590584e032c3be997620b217e44bd0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=12727
if lsof -i :$PORT; then
  echo "Port $PORT is in use. Killing the process using the port."
  lsof -ti :$PORT | xargs kill -9
else
  echo "Port $PORT is free."
fi

echo "PBS_NODEFILE = $(cat ${PBS_NODEFILE})"
echo "NODES_ARRAY = ${NODES_ARRAY[@]}"
echo "HEAD_NODE = ${HEAD_NODE}"
echo "HEAD_NODE_IP = ${HEAD_NODE_IP}"
echo "HOST_LIST = ${HOST_LIST}"
echo "NNODES = ${NNODES}"
echo "NGPUS = ${NGPUS}"
echo "NCPUS_PER_GPU = ${NCPUS_PER_GPU}"
echo "MASTER_ADDR:MASTER_PORT = $MASTER_ADDR:$MASTER_PORT"

# Path setup
HOME=/home/sukriti5
export PATH="/soft/perftools/darshan/darshan-3.4.4/bin:/opt/cray/pe/perftools/23.12.0/bin:/opt/cray/pe/papi/7.0.1.2/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/cray/pals/1.3.4/bin:/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin:/opt/cray/pe/mpich/8.1.28/bin:/opt/cray/pe/craype/2.7.30/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/opt/cray/pe/bin"
export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${HOME}/.local/bin:${HOME}/miniconda3/bin:${HOME}/.local/bin:${HOME}/bin:$PATH"
echo "PATH = $PATH"

# modules
module use /soft/modulefiles
module load conda
module load cudatoolkit-standalone/11.8.0
module load cray-mpich
conda activate /lus/eagle/projects/DemocAI/sukriti/envs/sdxl

huggingface-cli login --token $HF_TOKEN

cd /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/text_to_image

# DeepSpeed integration
export ACCELERATE_CONFIG_FILE='/lus/eagle/projects/DemocAI/sukriti/pbs_scripts/accelerate_deepspeed_mn.yaml'

# Debug: Print environment variables
echo "MASTER_ADDR = $MASTER_ADDR"
echo "MASTER_PORT = $MASTER_PORT"
echo "PBS_NODENUM= $PBS_NODENUM"

# Launch training
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
  --num_processes $NGPUS --num_machines $NNODES --machine_rank $PBS_NODENUM \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/text_to_image/train_TTI_sdxllora_PPcompare.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_dir='/lus/eagle/projects/DemocAI/common-datasets/cc12m_gemini' \
  --resolution=1024 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=2e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="/lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/sd-naruto-model-lora-sdxl-ppv2" \
  --validation_prompt="cute dragon creature" --report_to="wandb" \
  #--push_to_hub
