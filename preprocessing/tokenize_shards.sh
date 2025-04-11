#!/bin/bash

#SBATCH --account=project_465001890
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=standard


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
source ${HOME}/.bashrc
export EBU_USER_PREFIX=/projappl/project_465001925/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

# process arguments
## input and output directories
INPUT_PATHS=${1}
OUTPUT_PATHS=${2}
TOKENIZER_PATH=${3}
OTHER_ARGS=${@:4}

export WORLD_SIZE=$SLURM_NTASKS
echo "Node name${SLURMD_NODENAME}"
echo "Node list ${SLURM_NODELIST}"
echo "Kernel version"
uname -a
# run the script
echo "Running tokenize_shards.py --input_files=${INPUT_PATHS} --output_files=${OUTPUT_PATHS} --tokenizer_path=${TOKENIZER_PATH} ${OTHER_ARGS}"
srun singularity exec $SIF python3 tokenize_shards.py --input_files=${INPUT_PATHS} --output_files=${OUTPUT_PATHS} --tokenizer_path=${TOKENIZER_PATH} ${OTHER_ARGS}
