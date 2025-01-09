#!/bin/bash

#SBATCH --job-name=HPLT_BERT
#SBATCH --account=project_465001386
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=mi250:8
#SBATCH --partition=standard-g
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=logs/bert-%j.out


mkdir -p workdir
wd=$(realpath workdir)
# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch --job-name "${1}-BERT" --output "logs/${1}-bert-%j.out" "$0" "$@"
    exit
fi

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10


set -euo pipefail

LANGUAGE=${1}
PROCESSED_DIR=${2}

# no need for the long path if slurm is run with chdir
CMD=" \
    train.py \
    --language $LANGUAGE \
    --input_dir $PROCESSED_DIR \
    --output_dir $PROCESSED_DIR \
"

# Bind masks from Samuel Antao
c=fe

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ ! -d $wd/cray-deps ] ; then
  rm -rf $wd/cray-deps
  mkdir $wd/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B /opt/cray:/opt/cray \
    -B $wd/cray-deps:/opt/cray-deps \
    -B $wd:/workdir \
    $SIF \
    ~/HPLT-WP4/encoder-only/launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
