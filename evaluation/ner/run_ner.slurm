#!/bin/bash
#SBATCH --job-name=bert_ner
#SBATCH --account=project_465001890
#SBATCH --partition=small-g
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

source ${HOME}/.bashrc
export EBU_USER_PREFIX=/projappl/project_465001925/software/
module --quiet purge
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

MODEL=${1}  # BERT model (e.g. /cluster/projects/nn9851k/models/norlm/norbert3-small/)
DATASET=${2}  # wikiann/no
OUT=${3}  # where to save the models and results

echo ${MODEL}
echo ${DATASET}
echo ${OUT}

srun singularity exec $SIF python3 ner_bert.py --model_name_or_path ${MODEL} --dataset_name ${DATASET} --output_dir ${OUT} --trust_remote_code True
