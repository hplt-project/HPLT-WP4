#!/bin/bash

#SBATCH --account=project_465001386
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
source ${HOME}/.bashrc
export EBU_USER_PREFIX=/projappl/project_465001384/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

# process arguments
## input and output directories
INPUT_DIR=${1}
OUTPUT_DIR=${2}
OTHER_ARGS=${@:3}

# run the script
echo "Running train_tokenizer.py input_dir ${INPUT_DIR} output_dir ${OUTPUT_DIR} ${OTHER_ARGS}"
srun singularity exec $SIF python3 train_tokenizer.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} ${OTHER_ARGS}
