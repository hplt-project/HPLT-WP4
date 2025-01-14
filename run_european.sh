#!/bin/bash


for lang in tur_Latn fin_Latn fao_Latn lit_Latn lij_Latn ron_Latn gle_Latn tat_Cyrl lvs_Latn ltg_Latn ltz_Latn rus_Cyrl pol_Latn est_Latn nld_Latn slv_Latn nob_Latn nno_Latn ukr_Cyrl hye_Armn slk_Latn kat_Geor eus_Latn gla_Latn hun_Latn mlt_Latn
do
     rename_lang=${lang:0:3}${lang:4:1}
     echo $rename_lang
     ./schedule.sh $rename_lang /scratch/project_465001386/hplt-2-0-full  /scratch/project_465001386/hplt-2-0-output 512 0.0 --first_file_only
done
