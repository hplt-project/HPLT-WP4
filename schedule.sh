#!/bin/bash

#SBATCH --job-name=SCHEDULER
#SBATCH --account=project_465000498
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH --output=logs/schedule-%j.out


if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch --job-name "${1}-SCHEDULE" --output "logs/${1}-schedule-%j.out" "$0" "$@"
    exit
fi

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
LANGUAGE=${1}
INPUT_DIR="/scratch/project_465000498/one/cleaned/${LANGUAGE}"
OUTPUT_DIR="/scratch/project_465000498/processed_data/${LANGUAGE}"

## shard size in MB, default 256
SHARD_SIZE_MB=${2:-512}
SAMPLE_POWER=${3:-0.0}

# run the script
echo "Running schedule.py --language ${LANGUAGE} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --shard_size_mb ${SHARD_SIZE_MB} --sample_power ${SAMPLE_POWER}"
python3 schedule.py --language ${LANGUAGE} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --shard_size_mb ${SHARD_SIZE_MB} --sample_power ${SAMPLE_POWER}
