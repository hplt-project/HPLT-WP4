for lang in cmn_Hans eng_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn swe_Latn vie_Latn
    do
        sbatch rewrite_olivia.sh --input_dir /cluster/work/projects/nn9851k/mariiaf/hplt/ --folder tokenized_shards --num_files 8 --lang $lang --total 0.0
    done