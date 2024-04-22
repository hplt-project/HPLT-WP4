#! /bin/bash

for i in hplt*/
do
    echo ${i}
    python3 upload.py HPLT/${i%*/} ${i}
done