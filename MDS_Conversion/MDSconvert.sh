#!/bin/bash
#PBS -N mds_conversion
#PBS -l filesystems=home:eagle
#PBS -l select=2
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -A DemocAI
#PBS -o /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_out.log
#PBS -e /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_err.log

# Clear log files before the job starts
: > /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_out.log
: > /lus/eagle/projects/DemocAI/sukriti/diffusion_experiments/logs/lora_run01_err.log


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
export NUM_PROCESSES=64

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

cd /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/text_to_image

# Debug: Print environment variables
echo "MASTER_ADDR = $MASTER_ADDR"
echo "MASTER_PORT = $MASTER_PORT"
echo "PBS_NODENUM= $PBS_NODENUM"

mpiexec -np $NUM_PROCESSES python3 -u /lus/eagle/projects/DemocAI/sukriti/diffusers/examples/text_to_image/save_mds_mosaic.py 

