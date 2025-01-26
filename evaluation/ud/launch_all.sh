#!/bin/bash
for lang in /scratch/project_465001386/hplt-2-0-output/hplt_hf_models/*
  do
    lang_only=$(basename "$lang")
    echo $lang_only
    if [ ! -f  /scratch/project_465001386/hplt-2-0-output/results/${lang_only}_hplt.jsonl ]; then
      echo "Running ${lang_only}"
      sbatch --job-name ${lang_only}-UD  train.sh --language $lang_only
    fi
  done