# this script is used to shard the data into smaller files
# it takes in the input directory, output directory, and the shard size
# the input directory is the directory containing N the jsonl files
# the shard size is the max number of documents per shard
# the output directory is the directory where the K sharded files for training and an extra validation file will be stored
# K is calculated to be a power of 2

import argparse
import os
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shard_size', type=int, required=True, help='Max number of documents per shard')
    return parser.parse_args()


def count_number_of_documents(input_dir):
    num_documents = 0
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue

        num_lines = sum(1 for _ in open(os.path.join(input_dir, filename), "r"))
        num_documents += num_lines
    
    return num_documents


def shard(input_dir, output_dir, shard_size):
    number_of_documents = count_number_of_documents(input_dir)
    number_of_shards = 2 ** math.ceil(math.log(number_of_documents / shard_size, 2))

    print(f"Number of documents: {number_of_documents}")
    print(f"Number of shards: {number_of_shards}")

    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # remove any existing files in the output directory
    for filename in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, filename))
    
    # open all shard files
    shard_files = [
        open(os.path.join(output_dir, f"train_{i}.jsonl"), "w")
        for i in range(number_of_shards)
    ]
    shard_files.append(open(os.path.join(output_dir, "validation_0.jsonl"), "w"))
    shard_index = 0

    # iterate through all input files
    for filename in os.listdir(input_dir):
        for line in open(os.path.join(input_dir, filename), "r"):
            shard_file = shard_files[shard_index % number_of_shards]
            shard_file.write(line)
            shard_index += 1

    # close all shard files
    for shard_file in shard_files:
        shard_file.close()


if __name__ == "__main__":
    args = parse_args()
    shard(args.input_dir, args.output_dir, args.shard_size)
