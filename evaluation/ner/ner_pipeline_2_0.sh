for lang in /scratch/project_465001386/hplt-2-0-output/hplt_hf_models/*
  do
    lang_only=$(basename "$lang")
    echo $lang_only
    if [ ! -f  /scratch/project_465001386/hplt-2-0-output/logs/ner_${lang_only}/ner_results.tsv ]; then
      echo "Running ${lang_only}"
      sbatch -J ner-${lang_only} run_ner.slurm /scratch/project_465001386/hplt_hf_models/${lang_only}/ wikiann/${lang_only} /scratch/project_465001386/hplt-2-0-output/logs/ner_${lang_only}
    fi
  done