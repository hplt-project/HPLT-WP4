#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH --account=project_465001890
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH --output=/scratch/project_465001890/hplt-2-0-output/logs/baseline-%j.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
source ${HOME}/.bashrc
export EBU_USER_PREFIX=/projappl/project_465001925/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

# run the script
for lang in ellG hebH indL vieL
  do
    echo lang
    srun singularity exec $SIF python3 create_random_baseline.py --language $lang
  done