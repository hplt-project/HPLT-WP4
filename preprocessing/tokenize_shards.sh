#!/bin/bash

#SBATCH --account=project_465000498
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=small


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# Load the virtual environment
source /project/project_465000144/pytorch_1.13.1/bin/activate

# process arguments
## input and output directories
INPUT_PATHS=${1}
OUTPUT_PATHS=${2}
TOKENIZER_PATH=${3}

export WORLD_SIZE=$SLURM_NTASKS

# run the script
echo "Running tokenize_shards.py --input_files=${INPUT_PATHS} --output_files=${OUTPUT_PATHS} --tokenizer_path=${TOKENIZER_PATH}"
srun -W 0 python3 tokenize_shards.py --input_files=${INPUT_PATHS} --output_files=${OUTPUT_PATHS} --tokenizer_path=${TOKENIZER_PATH}
