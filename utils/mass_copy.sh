#! /bin/bash

for i in hplt*/
do
    echo ${i}
    code=`echo ${i} | cut -d "_" -f 4`
    code=${code%*/}
    cp bert_base_en/README.md ${i}
    sed -i "0,/- en/s//- $code/" ${i}README.md
done