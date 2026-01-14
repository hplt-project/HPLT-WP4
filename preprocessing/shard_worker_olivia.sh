#!/bin/bash

#SBATCH --account=nn10029k
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"
# process arguments
## input and output directories
INPUT_PATHS=${1}
OUTPUT_DIR=${2}
SHARDS=${3}
SAMPLE_POWER=${4:-"0.0"}
OTHER_ARGS=${@:5}

# run the script
echo "Running shard_worker.py --input_paths ${INPUT_PATHS} --output_dir ${OUTPUT_DIR} --shards ${SHARDS} --sample_power ${SAMPLE_POWER} ${OTHER_ARGS}"

srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/,/nird/datalake/NS8112K/public/three/sorted/:/nird/datalake/NS8112K/public/three/sorted/ $SIF python3 shard_worker.py --input_files ${INPUT_PATHS} --output_dir ${OUTPUT_DIR} --shards ${SHARDS} --sample_power ${SAMPLE_POWER} ${OTHER_ARGS}
