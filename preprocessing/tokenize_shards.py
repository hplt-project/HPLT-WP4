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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, required=True)
    parser.add_argument('--output_files', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    return parser.parse_args()


def limit_repetitions(s):
    return re.sub(r'(\S)(\1{7,})', lambda m: m.group(1) * 8, s)


def tokenize(tokenizer, text):
    text = text.rstrip("\n")
    text = limit_repetitions(text)
    ids = tokenizer.encode(line, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.int16)

    return ids


if __name__ == "__main__":
    args = parse_args()

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

    # tokenize file
    tokenized_documents = []
    n_subwords = 0
    for line in tqdm(open(input_file)):
        document = json.loads(line)
        tokenized_document = tokenize(tokenizer, document)
        tokenized_documents.append(tokenized_document)
        n_subwords += len(tokenized_document)

    # save the tokenized documents
    with gzip.GzipFile(output_file, 'wb') as f:
        torch.save(tokenized_documents, f)

    # remove the original file
    os.remove(input_file)

    print(f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total")
