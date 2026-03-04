#!/bin/bash
for lang in nno_Latn mlt_Latn gle_Latn eng_Latn bel_Cyrl ukr_Cyrl cmn_Hans vie_Latn por_Latn cat_Latn tam_Taml ind_Latn pol_Latn spa_Latn jpn_Jpan ita_Latn nld_Latn hye_Armn fin_Latn tur_Latn eus_Latn ekk_Latn isl_Latn glg_Latn heb_Hebr slk_Latn lvs_Latn ell_Grek slv_Latn lit_Latn kor_Hang ces_Latn dan_Latn hun_Latn ron_Latn cym_Latn deu_Latn fra_Latn nob_Latn swe_Latn
    do
        sbatch --job-name ${lang}-UD --output=/cluster/work/projects/nn9851k/mariiaf/hplt/logs/${lang}-UD--%j.out olivia.slurm --language $lang \
                            --treebank_path /cluster/work/projects/nn9851k/mariiaf/hplt/ud-treebanks-v2.13/ \
                            --models_path /cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models/ \
                            --results_path /cluster/work/projects/nn9851k/mariiaf/hplt/ud_results/ \
                            --checkpoints_path /cluster/work/projects/nn9851k/mariiaf/hplt/ud_checkpoints/
    done
