import os

# List of input folders
input_folders = [
    "cc12m_part0", "cc12m_part4", "cc12m_part8", "commonpool_node0_part1", "commonpool_node0_part5", "commonpool_node3_part0", "redcaps_part3", "redcaps_partd",
    "cc12m_part1", "cc12m_part5", "cc12m_part9", "commonpool_node0_part2", "commonpool_node0_part6", "commonpool_node4_part0", "redcaps_part0", "redcaps_part4",
    "cc12m_part2", "cc12m_part6", "cc12m_partd", "commonpool_node0_part3", "commonpool_node1_part0", "commonpool_node5_part0", "redcaps_part1", "redcaps_part5",
    "cc12m_part3", "cc12m_part7", "commonpool_node0_part0", "commonpool_node0_part4", "commonpool_node2_part0", "commonpool_noded_part0", "redcaps_part2", "redcaps_part6"
]

# SLURM script template
slurm_script_template = """#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

module unload cuda/12.4.1
module load cuda/12.1.1
source activate /fs/cml-projects/yet-another-diffusion/sd3m-diffusion/env/sd3-medium-lora

export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export HF_TOKEN="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"

python /fs/cml-projects/yet-another-diffusion/sd3m-diffusion/nexus/precompute_sd3_embeddings.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
  --tar_path "/fs/cml-projects/yet-another-diffusion/pixelprose-shards/{input_folder}/{input_folder}_*.tar" \
  --output_dir "/fs/cml-projects/yet-another-diffusion/pixelprose_sd3_latents/{input_folder}/" \
  --batch_size 32 \
  --num_workers 0 \
  --max_length 77
"""

# Create a directory to store the generated scripts
os.makedirs("slurm_scripts", exist_ok=True)

# Generate a SLURM script for each input folder
for folder in input_folders:
    script_content = slurm_script_template.format(input_folder=folder)
    script_filename = f"slurm_scripts/job_{folder}.sh"
    
    with open(script_filename, "w") as script_file:
        script_file.write(script_content)
    
    print(f"Generated SLURM script: {script_filename}")

print("All SLURM scripts have been generated in the 'slurm_scripts' directory.")