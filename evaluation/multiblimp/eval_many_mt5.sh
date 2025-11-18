#!/bin/sh
for lang in sqi arb gle est lav hye bul cat ces cym dan nld ell eus fao fin fra glg kat deu heb hun isl ita kmr lit pol mkd por ron rus slk slv spa swe tam tur ukr
do
    model_name=google/mt5-base
    echo $lang
    sbatch --output=${lang}-google-base.out eval_model.sh --model_name ${model_name} --data_filename ${lang}/data.tsv --mask_1 "▁<extra_id_0>" --mask_2 "▁<extra_id_1>"
done