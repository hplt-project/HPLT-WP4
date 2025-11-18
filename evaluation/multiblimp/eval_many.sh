#!/bin/sh
for lang in eng_Latn ces_Latn cat_Latn eus_Latn fin_Latn fra_Latn glg_Latn spa_Latn ukr_Cyrl kat_Geor hye_Armn bul_Cyrl deu_Latn fao_Latn cym_Latn dan_Latn nld_Latn ell_Grek heb_Hebr hun_Latn isl_Latn ita_Latn kmr_Latn lit_Latn pol_Latn mkd_Cyrl por_Latn ron_Latn slk_Latn slv_Latn  swe_Latn tam_Taml tur_Latn 
do
    model_name=/scratch/project_465002259/hplt-3-0-t5/hf_models/${lang}_31250
    lang=${lang:0:3} 
    echo $lang
    sbatch --output=${lang}-our-spaces.out eval_model.sh --model_name ${model_name} --data_filename ${lang}/data.tsv --mask_1 "[MASK_1]" --mask_2 "[MASK_2]"
done