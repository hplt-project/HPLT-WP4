#!/bin/bash

# Job name:
#SBATCH --job-name=ud
#
# Project:
#SBATCH --account=nn9851k
#
# Wall time limit:
#SBATCH --time=3:00:00
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module --force swap StdEnv Zen2Env
module use -a /cluster/shared/nlpl/software/eb/etc/all/

module load nlpl-accelerate/0.27.2-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8 # for mt5-base, mdeberta
module load nlpl-wandb/0.15.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.38.2-foss-2022b-Python-3.10.8
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8
module load nlpl-nlptools/02-foss-2022b-Python-3.10.8

module list    # For easier debugging

cd ~/HPLT-WP4/evaluation/ner
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

MODEL=${1}  # BERT model (e.g. /cluster/projects/nn9851k/models/norlm/norbert3-small/)
DATASET=${2}  # wikiann/no
OUT=${3}  # where to save the models and results

echo ${MODEL}
echo ${DATASET}
echo ${OUT}

python3 ner_bert.py --model_name_or_path ${MODEL} --dataset_name ${DATASET} --output_dir ${OUT} --trust_remote_code True
