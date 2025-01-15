#!/bin/bash

LANGS=${@}
for lang in LANGS
do
     rename_lang=${lang:0:3}${lang:4:1}
     echo $rename_lang
     ./schedule.sh $rename_lang /scratch/project_465001386/hplt-2-0-full  /scratch/project_465001386/hplt-2-0-output 512 0.0 --first_file_only
done
