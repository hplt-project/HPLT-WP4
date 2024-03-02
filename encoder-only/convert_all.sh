#! /bin/bash

while read l; do
	echo ${l}
	python3 convert_intermediate_to_hf.py --language ${l}
done < 75langs.txt

