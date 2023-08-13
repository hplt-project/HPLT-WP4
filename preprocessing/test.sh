export TOKENIZERS_PARALLELISM=false

python3 shard.py --input_dir ../data/test --output_dir ../data/shard_test --shard_size 10
python3 train_tokenizer.py --input_dir ../data/shard_test --validation_file ../data/shard_test/validation_0.jsonl.gz --tokenizer_path ../tokenizer_test.json --vocab_size 128 --do_calculate_stats
python3 tokenize_shards.py --input_dir ../data/shard_test --output_dir ../data/tokenized_test --tokenizer_path ../tokenizer_test.json