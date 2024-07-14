NODES_ARRAY=($(cat "${PBS_NODEFILE}" | sort | uniq))
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(getent hosts $HEAD_NODE | awk 'NR==1 { print $1 }')
HOST_LIST=$(IFS=,; echo "${NODES_ARRAY[*]}")
NNODES= 1 #$(wc -l < $PBS_NODEFILE)
NGPUS_PER_NODE=4
NGPUS=$((NGPUS_PER_NODE * NNODES))
NCPUS_PER_GPU=8


#DeepSpeed formatted hostfile
DEEPEED_HOSTFILE="/lus/eagle/projects/DemocAI/sukriti/pbs_scripts/deepspeed_hostfile"
# : > $DEEPEED_HOSTFILE
# for NODE in "${NODES_ARRAY[@]}"; do
#   echo "${NODE} slots=${NGPUS_PER_NODE}" >> $DEEPEED_HOSTFILE
# done

export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=12727  # From Kai's code
export HF_TOKEN="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"
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export WANDB_API_KEY="fdd9c522e7590584e032c3be997620b217e44bd0"
export WANDB_MODE=offline
export WORLD_SIZE=$NGPUS  #should be 8
export NCCL_DEBUG=INFO
export INSTANCE_DIR="/lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/sd3images/dog"
export OUTPUT_DIR="/lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/SD3-M"


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


#huggingface-cli login --token $HF_TOKEN

cd /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth

# DeepSpeed integration
export ACCELERATE_CONFIG_FILE='/lus/eagle/projects/DemocAI/sukriti/pbs_scripts/deep_speed_sd3.yaml'

# Set CUDA_VISIBLE_DEVICES for each process
#export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $((NGPUS_PER_NODE - 1)))
#ADDING SHARED MEM CLEAR UP (Doesn't work when I add to this script :()

#--hostfile $DEEPEED_HOSTFILE


# deepspeed  --num_nodes 1 \
#   --num_gpus $NGPUS_PER_NODE /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/train_dreambooth_lora_sd3.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --mixed_precision="fp16" \
#   --instance_prompt="a photo of sks dog" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=8 \
#   --learning_rate=1e-5 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --validation_prompt="A photo of sks dog in a bucket" \
#   --validation_epochs=25 \
#   --seed="0" 
#   #--report_to="wandb" 
# #  --dataloader_num_workers=8 \


## Full finetune
# python3 train_sd3_finetune.py \
# --dataset_name="/lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/dog/" \
# --pretrained_model_name_or_path=$MODEL_NAME \
# --output_dir=$OUTPUT_DIR \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=1 \
# --sample_batch_size=1 \
# --gradient_accumulation_steps=2 \
# --learning_rate=0.0001 \
# --report_to="wandb" \
# --lr_scheduler="constant" \
# --lr_warmup_steps=0 \
# --max_train_steps=2000 \
# --weighting_scheme="logit_normal" \
# --seed="42" \
# --gradient_checkpointing \
# --prior_generation_precision="fp16" \
# --caption_column="caption" \
# --dataloader_num_workers=0


deepspeed  --num_nodes 1 \
--num_gpus $NGPUS_PER_NODE /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/train_sd3_finetune.py \
--dataset_name="/lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/dog/" \
--pretrained_model_name_or_path=$MODEL_NAME \
--output_dir=$OUTPUT_DIR \
--mixed_precision="fp16" \
--resolution=512 \
--train_batch_size=4 \
--sample_batch_size=4 \
--gradient_accumulation_steps=2 \
--learning_rate=0.0001 \
--report_to="wandb" \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps=2000 \
--weighting_scheme="logit_normal" \
--seed="42" \
--gradient_checkpointing \
--prior_generation_precision="fp16" \
--caption_column="caption" \
--dataloader_num_workers=0