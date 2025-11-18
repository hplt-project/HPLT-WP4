#!/bin/bash
#SBATCH --job-name=multiblimp
#SBATCH --account=project_465002259
#SBATCH --partition=small-g
#SBATCH --gpus=1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=7

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
source ${HOME}/.bashrc
export EBU_USER_PREFIX=/projappl/project_465001925/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
echo ${@}
srun singularity exec $SIF python3 eval_model.py ${@}
