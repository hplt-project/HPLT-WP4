#!/bin/bash

# Project:
#SBATCH --account=nn10029k
#
# Wall time limit:
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=accel
#SBATCH --mem=0

echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"

SIF="/cluster/work/projects/nn9851k/mariiaf/hplt-container/pytorch2.7_cu2.9_py3.12_arm_ner_ud.sif"

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

srun apptainer exec --nv -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 train.py "$@"
