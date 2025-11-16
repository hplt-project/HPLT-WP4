#!/bin/bash

#SBATCH --account=project_465002259
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small


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
OUTPUT_DIR=${2}
SHARDS=${3}
SAMPLE_POWER=${4:-"0.0"}
OTHER_ARGS=${@:5}

# run the script
echo "Running shard_worker.py --input_paths ${INPUT_PATHS} --output_dir ${OUTPUT_DIR} --shards ${SHARDS} --sample_power ${SAMPLE_POWER} ${OTHER_ARGS}"
srun singularity exec $SIF python3 shard_worker.py --input_files ${INPUT_PATHS} --output_dir ${OUTPUT_DIR} --shards ${SHARDS} --sample_power ${SAMPLE_POWER} ${OTHER_ARGS}
