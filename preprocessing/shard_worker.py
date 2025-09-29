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
    parser.add_argument('--docs_to_pick', type=int, default=0)
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


def shard_sample(input_files, output_dir, shards, args, create_validation=False, sample_power=0.0):

    random.seed(42)

    # open all shard files
    shard_files = [
        gzip.open(os.path.join(output_dir, f"train_{i:05d}.jsonl.gz"), "wt")
        for i in shards
    ]

    if create_validation:
        validation_file = gzip.open(os.path.join(output_dir, "validation.jsonl.gz"), "wt")

    n_validation_documents = 10_000
    n_processed_train_documents = 0
    n_processed_val_documents = 0
    n_rejected_documents = 0

    # iterate through all input files
    for filename in input_files:
        # open/decompress every .json.zst file
        dctx = zstd.ZstdDecompressor()     
        with open(filename, "rb") as f:
            print(filename, flush=True)
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for num_samples, _ in enumerate(tqdm(text_stream)):
                    continue       
        print(num_samples, flush=True)
        docs_to_pick = min(args.docs_to_pick, num_samples)
        indices_to_pick = set(random.sample([i for i in range(num_samples + 1)], k=args.docs_to_pick))
        print(len(indices_to_pick), flush=True)
        with open(filename, "rb") as f:
            print(filename, flush=True)
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for i, line in enumerate(tqdm(text_stream)):
                    if (n_processed_train_documents == docs_to_pick) and ((not create_validation) or (n_processed_val_documents == n_validation_documents)):
                        break
                    if i in indices_to_pick:
                        line = json.loads(line)
                        document = line["text"].strip()
                        if len(document) == 0:
                            indices_to_pick.add(random.choice([j for j in range(i + 1, num_samples + 1) if j not in indices_to_pick]))
                            continue

                        if sample_power > 0.0:
                            scores = line["scores"]
                            scores = [float(score) for score in scores]
                            mean_score = mean(scores)

                            if random.random() > math.pow(mean_score + 0.2, sample_power):
                                n_rejected_documents += 1
                                indices_to_pick.add(random.choice([j for j in range(i + 1, num_samples + 1) if j not in indices_to_pick]))
                                continue

                        if n_processed_train_documents == 0:
                            print(f"\nFirst document: {document}\n\n", flush=True)
                        shard_file = shard_files[n_processed_train_documents % len(shard_files)]
                        shard_file.write(json.dumps(document) + "\n")
                        n_processed_train_documents += 1
                        indices_to_pick.remove(i)
                    else:
                        # write to validation file if shard_index < validation_size
                        if create_validation and (n_processed_val_documents < n_validation_documents):
                            line = json.loads(line)
                            document = line["text"].strip()
                            if len(document) == 0:
                                continue
                            validation_file.write(json.dumps(document) + "\n")
                            n_processed_val_documents += 1
    assert n_processed_train_documents == docs_to_pick
    # close all shard files
    for shard_file in shard_files:
        shard_file.close()
    if create_validation:
        validation_file.close()

    if sample_power > 0.0:
        print(f"Rejected {n_rejected_documents} train documents ({n_rejected_documents / (n_processed_train_documents + n_rejected_documents) * 100.0:.2f}%)", flush=True)

 
if __name__ == "__main__":
    args = parse_args()

    args.input_files = args.input_files.split(",")
    args.shards = [int(shard) for shard in args.shards.split(",")]

    if args.docs_to_pick:
        shard_sample(args.input_files, args.output_dir, args.shards, args, args.create_validation, args.sample_power)
    else:
        shard(args.input_files, args.output_dir, args.shards, args.create_validation, args.sample_power)
