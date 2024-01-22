import argparse
import os
import io
import gzip
import zstandard as zstd
import math
import json
from tqdm import tqdm
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    return parser.parse_args()


def calculate(input_file, output_file):
    mean_scores = []

    # open/decompress every .json.zst file
    dctx = zstd.ZstdDecompressor()
    with open(input_file, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in tqdm(text_stream):
                scores = json.loads(line)["scores"]
                scores = [float(score) for score in scores]

                mean_scores.append(mean(scores))

    # close all shard files
    with open(output_file, "w") as f:
        f.write(json.dumps(mean_scores))


if __name__ == "__main__":
    args = parse_args()
    calculate(args.input_file, args.output_file)
