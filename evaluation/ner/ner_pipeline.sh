#! /bin/bash
#

while read LANG; do
  echo ${LANG}
  sbatch -J ner-${LANG} run_ner.slurm /scratch/project_465001386/hplt_hf_models/${LANG}/ wikiann/${LANG} hplt_${LANG}
done


