#!/bin/bash

# Change to the directory containing the job scripts
cd /fs/cml-projects/yet-another-diffusion/sd3m-diffusion/nexus/slurm_scripts

# Submit all job scripts
for script in job_*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done

echo "All jobs have been submitted."