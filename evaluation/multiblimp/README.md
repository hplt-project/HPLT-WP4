Adapted for encoder-decoders from [MultiBLIMP repo](https://github.com/jumelet/multiblimp/blob/main/scripts/lm_eval/eval_model.py)

```
module load LUMI PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404
singularity shell $SIF
pip install -r requirements.txt
```

Run single model

```
sbatch --output=eng.out eval_model.sh \
    --model_name /scratch/project_465002259/hplt-3-0-t5/hf_models/eng_Latn_31250 \
    --data_filename eng/data.tsv \
    --mask_1 "[MASK_1]" \
    --mask_2 "[MASK_2]"
```

Run many our models

`eval_many.sh` 

Beware of hardcoded path and there are not all our language codes there

Run many `google/mt5-base` models

`eval_many_mt5.sh`