#!/bin/bash

#SBATCH --job-name=CONVERT
#SBATCH --account=project_465001890
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH --output=/scratch/project_465001890/hplt-2-0-output/logs/convert-%j.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001925/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

INPUT_PATH=${1}
OUTPUT_DIR=${2}
ALL_CHECKPOINTS=${3} # 1 if convert all checkpoints
LANGS=${@:4}
for LANG in $LANGS
  do
    if [ $ALL_CHECKPOINTS != "1" ]; then
      srun singularity exec $SIF python3 convert_to_hf.py --input_model_directory $INPUT_PATH --output_model_directory $OUTPUT_DIR --language $LANG
    else
      srun singularity exec $SIF python3 convert_to_hf.py --input_model_directory $INPUT_PATH --output_model_directory $OUTPUT_DIR --language $LANG --all_checkpoints
    fi
  done
