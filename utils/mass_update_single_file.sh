#!/bin/bash
#SBATCH --job-name=upload
#SBATCH --account=nn10029k
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1750
#SBATCH --partition=normal
SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"
for lang in ltz_Latn nno_Latn mlt_Latn gle_Latn eng_Latn bel_Cyrl ukr_Cyrl cmn_Hans vie_Latn por_Latn cat_Latn tha_Thai tam_Taml ind_Latn pol_Latn spa_Latn jpn_Jpan ita_Latn rus_Cyrl nld_Latn hye_Armn fin_Latn als_Latn tur_Latn srp_Cyrl eus_Latn ekk_Latn isl_Latn glg_Latn slk_Latn lvs_Latn swh_Latn kmr_Latn tat_Cyrl ell_Grek slv_Latn mkd_Cyrl lit_Latn kor_Hang ces_Latn kat_Geor dan_Latn hun_Latn bul_Cyrl ara_Arab heb_Hebr ron_Latn cym_Latn deu_Latn
    do
        echo ${lang}
        srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 update_single_file.py /cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models/${lang}_31250/config.json --repo HPLT/hplt_gpt_bert_base_3_0_${lang}
    done