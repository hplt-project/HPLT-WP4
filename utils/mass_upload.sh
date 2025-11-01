#! /bin/bash

for path in /scratch/project_465002259/hplt-3-0-output/hf_models/*_31250
do
    echo ${path}
    lang=$(awk -v var="$path" 'BEGIN { print substr(var,length(var)-13, 8) }')
    echo ${lang}
    python3 upload.py HPLT/hplt_t5_base_3_0_${lang} ${path}
done