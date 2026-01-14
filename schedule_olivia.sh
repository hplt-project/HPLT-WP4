#!/bin/bash

#SBATCH --job-name=SCHEDULER
#SBATCH --account=nn10029k
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --output=/cluster/work/projects/nn9851k/mariiaf/hplt/logs/schedule-%j.out

if [ -z $SLURM_JOB_ID ]; then
    mkdir -p /cluster/work/projects/nn9851k/mariiaf/hplt/logs
    sbatch --job-name "${1}-SCHEDULE" --output "/cluster/work/projects/nn9851k/mariiaf/hplt/logs/${1}-schedule-%j.out" "$0" "$@"
    exit
fi

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# process arguments
## input and output directories
LANGUAGE=${1}
INPUT_DIR="${2}/${LANGUAGE}"
OUTPUT_DIR="${3}/${LANGUAGE}"

## shard size in MB, default 512
SHARD_SIZE_MB=${4:-512}
SAMPLE_POWER=${5:-0.0}
OTHER_ARGS=${@:6}

# run the script
echo "Running schedule.py --language ${LANGUAGE} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --shard_size_mb ${SHARD_SIZE_MB} --sample_power ${SAMPLE_POWER}"
python3 schedule.py --language ${LANGUAGE} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --shard_size_mb ${SHARD_SIZE_MB} --sample_power ${SAMPLE_POWER} ${OTHER_ARGS}
