NODES_ARRAY=($(cat "${PBS_NODEFILE}" | sort | uniq))
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(getent hosts $HEAD_NODE | awk 'NR==1 { print $1 }')
HOST_LIST=$(IFS=,; echo "${NODES_ARRAY[*]}")
NNODES=1
NGPUS_PER_NODE=4
NGPUS=$((NGPUS_PER_NODE * NNODES))
NCPUS_PER_GPU=8


#DeepSpeed formatted hostfile
DEEPEED_HOSTFILE="/lus/eagle/projects/DemocAI/sukriti/pbs_scripts/deepspeed_hostfile"
: > $DEEPEED_HOSTFILE
for NODE in "${NODES_ARRAY[@]}"; do
    echo "${NODE} slots=${NGPUS_PER_NODE}" >> $DEEPEED_HOSTFILE
done


#export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=12727  # From Kai's code
export HF_TOKEN="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"
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export WANDB_API_KEY="fdd9c522e7590584e032c3be997620b217e44bd0"
export WANDB_MODE=ONLINE
export WORLD_SIZE=$NGPUS  #should be 8
export NCCL_DEBUG=INFO
export INSTANCE_DIR="/lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/sd3images/dog"
export OUTPUT_DIR="/lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/SD3-Medium-basic"


export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"


export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PATH=/usr/bin:$PATH
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.2.2
export LD_LIBRARY_PATH=/soft/compilers/cudatoolkit/cuda-12.2.2/lib64:$LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

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
#echo "MASTER_ADDR:MASTER_PORT = $MASTER_ADDR:$MASTER_PORT"



# Path setup
HOME=/home/sukriti5
export PATH="/soft/perftools/darshan/darshan-3.4.4/bin:/opt/cray/pe/perftools/23.12.0/bin:/opt/cray/pe/papi/7.0.1.2/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/cray/pals/1.3.4/bin:/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin:/opt/cray/pe/mpich/8.1.28/bin:/opt/cray/pe/craype/2.7.30/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/opt/cray/pe/bin"
export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${HOME}/.local/bin:${HOME}/miniconda3/bin:${HOME}/.local/bin:${HOME}/bin:$PATH"
echo "PATH = $PATH"


# modules
module use /soft/modulefiles
module load conda
#module load cudatoolkit-standalone/11.8.0
module load cudatoolkit-standalone/12.2.2
module load cray-mpich
#conda activate /lus/eagle/projects/DemocAI/sukriti/envs/sdxl
conda activate sd3m


huggingface-cli login --token $HF_TOKEN
cd /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth

# DeepSpeed integration
export ACCELERATE_CONFIG_FILE='/lus/eagle/projects/DemocAI/sukriti/pbs_scripts/deep_speed_sd3.yaml'

# Set CUDA_VISIBLE_DEVICES for each process
export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $((NGPUS_PER_NODE - 1)))
#ADDING SHARED MEM CLEAR UP (Doesn't work when I add to this script :()


# LD_PRELOAD=/soft/compilers/cudatoolkit/cuda-12.2.2/lib64/libcurand.so \
# DS_DEBUG=1 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 deepspeed --hostfile $DEEPEED_HOSTFILE --num_nodes $NNODES --num_gpus $NGPUS_PER_NODE /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/train_sd3m_lora_mds.py \
#   --instance_data_dir="/lus/eagle/projects/DemocAI/common-datasets/PixelProseMDS/node3_new" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --output_dir=$OUTPUT_DIR \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=4 \
#   --sample_batch_size=4 \
#   --gradient_accumulation_steps=8 \
#   --learning_rate=0.00001 \
#   --report_to="wandb" \
#   --lr_scheduler="cosine_with_restarts" \
#   --lr_warmup_steps=100 \
#   --max_train_steps=10 \
#   --checkpointing_steps=10 \
#   --gradient_checkpointing \
#   --weighting_scheme="logit_normal" \
#   --seed=42 \
#   --prior_generation_precision="fp16" \
#   --caption_column="caption" \
#   --dataloader_num_workers=0 \
#   --adam_weight_decay=1e-03 \
#   --rank=2 \
#   --max_sequence_length=100 \
#   --max_grad_norm=1.0 \
#   --fp16_opt_level="O2" \
#   --zero_stage=2

LD_PRELOAD=/soft/compilers/cudatoolkit/cuda-12.2.2/lib64/libcurand.so \
DS_DEBUG=1 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 deepspeed --hostfile $DEEPEED_HOSTFILE --num_nodes $NNODES --num_gpus $NGPUS_PER_NODE /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/train_sd3m_lora_mds.py \
  --instance_data_dir="/lus/eagle/projects/DemocAI/common-datasets/PixelProseMDS/node3_new" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=4 \
  --sample_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=0.0001 \
  --report_to="wandb" \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=50 \
  --checkpointing_steps=5 \
  --gradient_checkpointing \
  --weighting_scheme="logit_normal" \
  --seed=42 \
  --prior_generation_precision="fp16" \
  --caption_column="caption" \
  --dataloader_num_workers=0 \
  --adam_weight_decay=1e-02 \
  --max_sequence_length=100 \
  --resume_from_checkpoint="latest" \
  --max_train_steps=10
  #--num_train_epochs=1
  # --validation_prompt="A high quality photo of a dog" \
  # --num_validation_images=4 \
  # --validation_epochs=50 \
   #--max_train_steps=10 \

  

# deepspeed --hostfile $DEEPEED_HOSTFILE --num_nodes $NNODES --num_gpus $NGPUS_PER_NODE /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/train_sd3m_lora_mds.py \
#   --instance_data_dir="/lus/eagle/projects/DemocAI/common-datasets/PixelProseMDS/node3_new" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --output_dir=$OUTPUT_DIR \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=8 \
#   --sample_batch_size=4 \
#   --gradient_accumulation_steps=4 \
#   --learning_rate=0.0001 \
#   --report_to="wandb" \
#   --lr_scheduler="cosine_with_restarts" \
#   --lr_warmup_steps=50 \
#   --max_train_steps=500 \
#   --checkpointing_steps=200 \
#   --weighting_scheme="logit_normal" \
#   --seed=42 \
#   --gradient_checkpointing \
#   --prior_generation_precision="fp16" \
#   --caption_column="caption" \
#   --dataloader_num_workers=0 \
#   --adam_weight_decay=1e-02 \
#   --validation_prompt="A high quality photo of a dog" \
#   --num_validation_images=4 \
#   --validation_epochs=50 \
#   --rank=4 \
#   --max_sequence_length=100



  # deepspeed --hostfile $DEEPEED_HOSTFILE --num_nodes $NNODES --num_gpus $NGPUS_PER_NODE /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/dreambooth/train_sd3_finetune.py \
  # --instance_data_dir="/lus/eagle/projects/DemocAI/common-datasets/PixelProseMDS/node3_new" \
  # --pretrained_model_name_or_path=$MODEL_NAME \
  # --output_dir=$OUTPUT_DIR \
  # --mixed_precision="fp16" \
  # --resolution=512 \
  # --train_batch_size=8 \
  # --sample_batch_size=4 \
  # --gradient_accumulation_steps=4 \
  # --learning_rate=0.0001 \
  # --report_to="wandb" \
  # --lr_scheduler="cosine_with_restarts" \
  # --lr_warmup_steps=50 \
  # --max_train_steps=500 \
  # --checkpointing_steps=200 \
  # --weighting_scheme="logit_normal" \
  # --seed=42 \
  # --gradient_checkpointing \
  # --prior_generation_precision="fp16" \
  # --dataloader_num_workers=0 \
  # --adam_weight_decay=1e-02 \
  # --random_flip \
  # --center_crop \
  # --validation_prompt="A high quality photo of a dog" \
  # --num_validation_images=4 \
  # --validation_epochs=50 \
  # --rank=4 \
  # --max_sequence_length=100


#LR: 5e-5, 2e-5
# train_batch_size -> highest
# atleast 1 epoch (intuitive #iter)
#10nodes on prod/ can try 20 nodes for 640 batch size-> 
#.1 to .5% for lr_warmup_steps (depends on max_train_steps)
#can be 1 radient_accumulation_steps. batch size increases 320 x 4. If i keep as 2, then inc nodes to 20ish. If kept at 1, inc nodes to 40.





# #!/bin/bash
# #PBS -N mds_conversion
# #PBS -l filesystems=home:eagle
# #PBS -l select=2
# #PBS -l walltime=12:00:00
# #PBS -q preemptable
# #PBS -A DemocAI
# #PBS -o /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_out.log
# #PBS -e /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_err.log

# # Clear log files before the job starts
# : > /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_out.log
# : > /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_err.log


# NODES_ARRAY=($(cat "${PBS_NODEFILE}" | sort | uniq))
# HEAD_NODE=${NODES_ARRAY[0]}
# HEAD_NODE_IP=$(getent hosts $HEAD_NODE | awk 'NR==1 { print $1 }')
# HOST_LIST=$(IFS=,; echo "${NODES_ARRAY[*]}")
# NNODES=$(wc -l < $PBS_NODEFILE)
# NGPUS_PER_NODE=4
# NGPUS=$((NGPUS_PER_NODE * NNODES))
# NCPUS_PER_GPU=16


# export MASTER_ADDR=$HEAD_NODE
# export MASTER_PORT=12727  # From Kai's code
# export NUM_PROCESSES=64

# echo "PBS_NODEFILE = $(cat ${PBS_NODEFILE})"
# echo "NODES_ARRAY = ${NODES_ARRAY[@]}"
# echo "HEAD_NODE = ${HEAD_NODE}"
# echo "HEAD_NODE_IP = ${HEAD_NODE_IP}"
# echo "HOST_LIST = ${HOST_LIST}"
# echo "NNODES = ${NNODES}"
# echo "NGPUS = ${NGPUS}"
# echo "NCPUS_PER_GPU = ${NCPUS_PER_GPU}"
# echo "MASTER_ADDR:MASTER_PORT = $MASTER_ADDR:$MASTER_PORT"

# # Path setup
# HOME=/home/sukriti5
# export PATH="/soft/perftools/darshan/darshan-3.4.4/bin:/opt/cray/pe/perftools/23.12.0/bin:/opt/cray/pe/papi/7.0.1.2/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/cray/pals/1.3.4/bin:/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin:/opt/cray/pe/mpich/8.1.28/bin:/opt/cray/pe/craype/2.7.30/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/opt/cray/pe/bin"
# export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${HOME}/.local/bin:${HOME}/miniconda3/bin:${HOME}/.local/bin:${HOME}/bin:$PATH"
# echo "PATH = $PATH"


# # modules
# module use /soft/modulefiles
# module load conda
# module load cudatoolkit-standalone/11.8.0
# module load cray-mpich
# conda activate /lus/eagle/projects/DemocAI/sukriti/envs/sdxl

# cd /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/text_to_image

# # Debug: Print environment variables
# echo "MASTER_ADDR = $MASTER_ADDR"
# echo "MASTER_PORT = $MASTER_PORT"
# echo "PBS_NODENUM= $PBS_NODENUM"

# mpiexec -np $NUM_PROCESSES python3 -u /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/text_to_image/save_mds_mosaic.py 

