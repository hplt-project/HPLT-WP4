#!/bin/bash

#SBATCH --job-name=SCHEDULER
#SBATCH --account=project_465001386
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH --output=/scratch/project_465001386/hplt-2-0-output/logs/schedule-%j.out # has no effect?


if [ -z $SLURM_JOB_ID ]; then
    mkdir -p /scratch/project_465001386/hplt-2-0-output/logs
    sbatch --job-name "${1}-SCHEDULE" --output "/scratch/project_465001386/hplt-2-0-output/logs/${1}-schedule-%j.out" "$0" "$@"
    exit
fi

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001384/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

# run the script
for lang in catL itaL
do
  echo "Running schedule.py"
  python3 start_from_tokenizing.py --language $lang
done