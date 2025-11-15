### How to train

```
sbatch train.sh \
    --input_dir /scratch/project_465002259/hplt-3-0-t5/ \
    --config_file configs/base.json \
    --max_steps 31250 \
    --batch_size 45 \
    --language mlt_Latn
```

### How to convert to the Huggingface format

```
sbatch convert_to_hf.sh /scratch/project_465002259/hplt-3-0-t5/ /scratch/project_465002259/hplt-3-0-t5/hf_models/ 1 mlt_Latn
```

1 means also convert all intermediate checkpoints

### How to run fine-tuning on WikiAnn NER

Our models:

```
sbatch --output=ast-latn.out  ner.sh \
    --model_path /scratch/project_465002259/hplt-3-0-t5/hf_models/ast_Latn_31250/ \
    --lang ast \
    --output_dir  /scratch/project_465002259/hplt-3-0-t5/t5-ner/
```

`google/mt5-base` has a larger checkpoint, so its batch size and learning rate had to be adjusted

```
sbatch --output=ast-google.out  ner.sh \
    --model_path google/mt5-base \
    --lang ast \
    --output_dir /scratch/project_465002259/hplt-3-0-output/t5-ner/ \
    --train_batch_size 5 \
    --learning_rate 5e-4
```