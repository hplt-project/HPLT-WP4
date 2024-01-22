import argparse
import os
import io
import gzip
import zstandard as zstd
import math
import json
from tqdm import tqdm
import random
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shards', type=str, required=True)
    parser.add_argument('--create_validation', action='store_true')
    parser.add_argument('--sample_power', type=float, default=0.0)
    return parser.parse_args()


def shard(input_files, output_dir, shards, create_validation=False, sample_power=0.0):

    random.seed(42)

    # open all shard files
    shard_files = [
        gzip.open(os.path.join(output_dir, f"train_{i:05d}.jsonl.gz"), "wt")
        for i in shards
    ]

    if create_validation:
        validation_file = gzip.open(os.path.join(output_dir, "validation.jsonl.gz"), "wt")

    n_validation_documents = 10_000
    n_processed_documents = 0
    n_rejected_documents = 0

    # iterate through all input files
    for filename in input_files:

        # open/decompress every .json.zst file
        dctx = zstd.ZstdDecompressor()
        with open(filename, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in tqdm(text_stream):
                    line = json.loads(line)
                    document = line["text"].strip()

                    if len(document) == 0:
                        continue

                    if sample_power > 0.0:
                        scores = line["scores"]
                        scores = [float(score) for score in scores]
                        mean_score = mean(scores)

                        if random.random() > math.pow(mean_score + 0.2, sample_power):
                            n_rejected_documents += 1
                            continue

                    if n_processed_documents == 0:
                        print(f"\nFirst document: {document}\n\n", flush=True)

                    # write to validation file if shard_index < validation_size
                    if create_validation and n_processed_documents < n_validation_documents:
                        validation_file.write(json.dumps(document) + "\n")
                    else:
                        shard_file = shard_files[n_processed_documents % len(shard_files)]
                        shard_file.write(json.dumps(document) + "\n")

                    n_processed_documents += 1

    # close all shard files
    for shard_file in shard_files:
        shard_file.close()
    if create_validation:
        validation_file.close()

    if sample_power > 0.0:
        print(f"Rejected {n_rejected_documents} documents ({n_rejected_documents / (n_processed_documents + n_rejected_documents) * 100.0:.2f}%)", flush=True)

 
if __name__ == "__main__":
    args = parse_args()

    args.input_files = args.input_files.split(",")
    args.shards = [int(shard) for shard in args.shards.split(",")]

    shard(args.input_files, args.output_dir, args.shards, args.create_validation, args.sample_power)
