#!/bin/bash

#SBATCH --job-name=HPLT_T5
#SBATCH --account=project_465001386
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=standard-g
#SBATCH --output=report/%j.out
#SBATCH --signal=B:TERM


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


# Load modules
source ${HOME}/.bashrc
export EBU_USER_PREFIX=/projappl/project_465001384/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617
module load rocm/5.2.3



export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2



##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK 

srun -W 0 python3 train.py "$@"
