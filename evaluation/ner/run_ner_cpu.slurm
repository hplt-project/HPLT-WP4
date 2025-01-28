#!/bin/bash
#SBATCH --job-name=bert_ner
#SBATCH --account=project_465000498
#SBATCH --partition=small
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16

source ${HOME}/.bashrc


module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1
module load rocm/5.2.3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

export PYTHONUSERBASE='/projappl/project_465000498/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

MODEL=${1}  # BERT model (e.g. /cluster/projects/nn9851k/models/norlm/norbert3-small/)
DATASET=${2}  # wikiann/no
OUT=${3}  # where to save the models and results

echo ${MODEL}
echo ${DATASET}
echo ${OUT}

python3 ner_bert.py --model_name_or_path ${MODEL} --dataset_name ${DATASET} --output_dir ${OUT} --trust_remote_code True
