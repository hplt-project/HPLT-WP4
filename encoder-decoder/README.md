### How to train

```
sbatch train.sh \
    --input_dir /scratch/project_465001890/hplt-2-0-output/train/swh-Latn/tokenized_shards/ \
    --name base \
    --config_file configs/base.json \
    --tokenizer_path /scratch/project_465001890/hplt-2-0-output/train/swh-Latn/tokenizer.json
```
