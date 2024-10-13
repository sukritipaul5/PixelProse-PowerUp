#!/bin/bash
#PBS -N PixelProse_FFT_sukriti
#PBS -l filesystems=home:eagle
#PBS -l select=25
#PBS -l walltime=06:00:00
#PBS -q prod
#PBS -A DemocAI
#PBS -o /lus/eagle/projects/DemocAI/sukriti/sd3m_deepspeed/logs/sd3m_fft_out_sukriti_25n.log
#PBS -e /lus/eagle/projects/DemocAI/sukriti/sd3m_deepspeed/logs/sd3m_fft_err_sukriti_25n.log


# Clear log files before the job starts
: > /lus/eagle/projects/DemocAI/sukriti/sd3m_deepspeed/logs/sd3m_fft_out_sukriti_25n.log
: > /lus/eagle/projects/DemocAI/sukriti/sd3m_deepspeed/logs/sd3m_fft_err_sukriti_25n.log

# Path setup
HOME=/home/sukriti5

# Get node information
NODES_ARRAY=($(cat "${PBS_NODEFILE}" | sort | uniq))
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(getent hosts $HEAD_NODE | awk 'NR==1 { print $1 }')
HOST_LIST=$(IFS=,; echo "${NODES_ARRAY[*]}")
NNODES=$(wc -l < $PBS_NODEFILE)
NGPUS_PER_NODE=4
NGPUS=$((NGPUS_PER_NODE * NNODES))
NCPUS_PER_GPU=8


#DeepSpeed formatted hostfile
DEEPEED_HOSTFILE="/lus/eagle/projects/DemocAI/sukriti/sd3m_deepspeed/pbs_scripts/deepspeed_hostfile"
: > $DEEPEED_HOSTFILE
cat $PBS_NODEFILE > $DEEPEED_HOSTFILE
sed -e 's/$/ slots=4/' -i $DEEPEED_HOSTFILE


# envs
export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=12727


export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"


# Set directories and configuration
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export OUTPUT_DIR="/lus/eagle/projects/DemocAI/sukriti/sd3m_deepspeed/outputs/sd3m_25nodes"
export HF_TOKEN="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"
export WANDB_API_KEY="fdd9c522e7590584e032c3be997620b217e44bd0"
export WORLD_SIZE=$NGPUS  

# Set compiler and CUDA paths
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PATH=/usr/bin:$PATH
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.2.2
export LD_LIBRARY_PATH=/soft/compilers/cudatoolkit/cuda-12.2.2/lib64:$LD_LIBRARY_PATH




# echo
echo "PBS_NODEFILE = $(cat ${PBS_NODEFILE})"
echo "NODES_ARRAY = ${NODES_ARRAY[@]}"
echo "HEAD_NODE = ${HEAD_NODE}"
echo "HEAD_NODE_IP = ${HEAD_NODE_IP}"
echo "HOST_LIST = ${HOST_LIST}"
echo "NNODES = ${NNODES}"
echo "NGPUS = ${NGPUS}"
echo "NCPUS_PER_GPU = ${NCPUS_PER_GPU}"
echo "MASTER_ADDR:MASTER_PORT = $MASTER_ADDR:$MASTER_PORT"

# src polaris path
export PATH="/soft/perftools/darshan/darshan-3.4.4/bin:/opt/cray/pe/perftools/23.12.0/bin:/opt/cray/pe/papi/7.0.1.2/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/cray/pals/1.3.4/bin:/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin:/opt/cray/pe/mpich/8.1.28/bin:/opt/cray/pe/craype/2.7.30/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/opt/cray/pe/bin"

# src user path
export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${HOME}/.local/bin:${HOME}/miniconda3/bin:${HOME}/.local/bin:${HOME}/bin:$PATH"

echo "PATH = $PATH"
module use /soft/modulefiles  
module load conda
module load cudatoolkit-standalone/12.2.2
module load cray-mpich
# # src conda path
# export PATH="${HOME}/miniconda3/bin:$PATH"

# DeepSpeed integration
export ACCELERATE_CONFIG_FILE='/lus/eagle/projects/DemocAI/vsingla/PixelProse-PowerUp/Vasu-ref/Polaris/deep_speed_sd3.yaml'
# Set CUDA_VISIBLE_DEVICES for each process
export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $((NGPUS_PER_NODE - 1)))


conda activate /lus/eagle/projects/DemocAI/sukriti/envs/sd3-medium-lora

huggingface-cli login --token $HF_TOKEN


echo "PATH=${PATH}" >> $HOME/.deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> $HOME/.deepspeed_env
echo "http_proxy=${http_proxy}" >> $HOME/.deepspeed_env
echo "https_proxy=${https_proxy}" >> $HOME/.deepspeed_env
echo "WANDB_API_KEY=${WANDB_API_KEY}" >> $HOME/.deepspeed_env
echo "HF_TOKEN=${HF_TOKEN}" >> $HOME/.deepspeed_env
echo "NCCL_DEBUG=INFO" >> $HOME/.deepspeed_env
echo "NCCL_SOCKET_IFNAME=hsn0" >> $HOME/.deepspeed_env

LAUNCH="deepspeed \
    --hostfile $DEEPEED_HOSTFILE \
    --num_nodes $NNODES \
    --num_gpus $NGPUS_PER_NODE \
    "
# Set up script and arguments
SCRIPT="/lus/eagle/projects/DemocAI/vsingla/PixelProse-PowerUp/Vasu-ref/Polaris_new/train_sd3m.py"

SCRIPT_ARGS="
  --pretrained_model_name_or_path $MODEL_NAME
  --output_dir $OUTPUT_DIR
  --mixed_precision bf16
  --resolution 512
  --train_batch_size 1
  --gradient_accumulation_steps 8 
  --learning_rate 4e-6
  --lr_scheduler cosine_with_restarts
  --lr_warmup_steps 1000
  --checkpointing_steps 100
  --gradient_checkpointing
  --weighting_scheme logit_normal
  --seed 42
  --use_custom_prompts True
  --dataloader_num_workers 0
  --adam_weight_decay 1e-02
  --max_sequence_length 77
  --max_train_steps 16000000
  --num_samples 16000000
  --num_validation_images 2
  --validation_step 50 "

echo "Launching command: $LAUNCH $SCRIPT $SCRIPT_ARGS"
$LAUNCH $SCRIPT $SCRIPT_ARGS

#1000, 50