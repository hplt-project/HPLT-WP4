#!/bin/bash

#SBATCH --account=nn10029k
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=7G

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# process arguments
## input and output directories
INPUT_PATHS=${1}
OUTPUT_PATHS=${2}
TOKENIZER_PATH=${3}
OTHER_ARGS=${@:4}

export WORLD_SIZE=$SLURM_NTASKS
echo "Node name ${SLURMD_NODENAME}"
echo "Node list ${SLURM_NODELIST}"
echo "Kernel version"
uname -a

echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"
# run the script
echo "Running tokenize_shards.py --input_files=${INPUT_PATHS} --output_files=${OUTPUT_PATHS} --tokenizer_path=${TOKENIZER_PATH} ${OTHER_ARGS}"
srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 tokenize_shards.py --input_files=${INPUT_PATHS} --output_files=${OUTPUT_PATHS} --tokenizer_path=${TOKENIZER_PATH} ${OTHER_ARGS}
