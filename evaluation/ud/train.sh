#!/bin/bash

#SBATCH --job-name=UD_EVAL
#SBATCH --account=project_465000498
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=report/%j.out


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1

source /project/project_465000144/pytorch_1.13.1/bin/activate

export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

export TOKENIZERS_PARALLELISM=false
python3 train.py "$@"
