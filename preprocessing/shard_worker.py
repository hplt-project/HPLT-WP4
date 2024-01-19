import argparse
import os
import io
import gzip
import zstandard as zstd
import math
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shards', type=str, required=True)
    parser.add_argument('--create_validation', action='store_true')
    return parser.parse_args()


def shard(input_files, output_dir, shards, create_validation=False):

    # open all shard files
    shard_files = [
        gzip.open(os.path.join(output_dir, f"train_{i:05d}.jsonl.gz"), "wt")
        for i in shards
    ]

    if create_validation:
        validation_file = gzip.open(os.path.join(output_dir, "validation.jsonl.gz"), "wt")

    n_validation_documents = 10_000
    n_processed_documents = 0

    # iterate through all input files
    for filename in input_files:

        # open/decompress every .json.zst file
        dctx = zstd.ZstdDecompressor()
        with open(filename, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in tqdm(text_stream):
                    document = json.loads(line)["text"].strip()

                    if len(document) == 0:
                        continue

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


if __name__ == "__main__":
    args = parse_args()

    args.input_files = args.input_files.split(",")
    args.shards = [int(shard) for shard in args.shards.split(",")]

    shard(args.input_files, args.output_dir, args.shards, args.create_validation)
