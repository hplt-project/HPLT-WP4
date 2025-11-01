#!/bin/bash


for model_name in /scratch/project_465002259/hplt-3-0-output/hf-3-0/*/modeling_nort5.py
    do
        echo ${model_name}
        cp ../encoder-decoder/huggingface_prototype/modeling_nort5.py ${model_name}
    done