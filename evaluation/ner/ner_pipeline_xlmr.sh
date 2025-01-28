#! /bin/bash
#

while read LANG; do
  echo ${LANG}
  sbatch -J ner-${LANG} run_ner.slurm /scratch/project_465000498/models/xlm-roberta-base wikiann/${LANG} xlmr_${LANG}
done


