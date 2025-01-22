# takes in the input directory, output directory, path to the tokenizer, and the max sequence length
# the input directory is the directory containing N sharded jsonl files
# the output directory is the directory where the each file is tokenized

from tokenizers import Tokenizer
import json
import os
import argparse
import re
from smart_open import open
import torch
import gzip
from tqdm import tqdm

from timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, required=True)
    parser.add_argument('--output_files', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    return parser.parse_args()


def limit_repetitions(s):
    return re.sub(r'(\S)(\1{7,})', lambda m: m.group(1) * 8, s)


def tokenize(tokenizer, text):
    text = text.rstrip()
    text = limit_repetitions(text)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.int16)

    return ids


if __name__ == "__main__":
    args = parse_args()

    # start a timer for 71 hours; if the timer runs out, the job will stop and the tokenized files will be saved
    timer = Timer(8 * 60 * 60)

    # load the tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])

    args.input_files = args.input_files.split(",")
    args.output_files = args.output_files.split(",")

    if rank >= len(args.input_files):
        exit(0)

    input_file = args.input_files[rank]
    output_file = args.output_files[rank]

    # check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping")
        exit(0)

    # tokenize file
    tokenized_documents = []
    n_subwords = 0

    if rank == 0:
        generator = tqdm(open(input_file, 'rt'))
    else:
        generator = open(input_file, 'rt')

    for i, line in enumerate(generator):
        try:
            document = json.loads(line)
        except:
            print(f"Error parsing line {i} of {input_file}, skipping")
            continue
        tokenized_document = tokenize(tokenizer, document)
        tokenized_documents.append(tokenized_document)
        n_subwords += len(tokenized_document)

        if i == 0:
            print("Example tokenized document:")
            print(document)
            for token in tokenized_document:
                print(tokenizer.decode([token]))
            print(flush=True)

        if not timer.has_time_remaining():
            print("No time remaining, breaking")
            break

        if (i + 1) % 10_000 == 0:
            with gzip.GzipFile(output_file, 'wb') as f:
                torch.save(tokenized_documents, f)

    # save the tokenized documents
    with gzip.GzipFile(output_file, 'wb') as f:
        torch.save(tokenized_documents, f)

    # remove the original file
    os.remove(input_file)

    print(
        f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total")
