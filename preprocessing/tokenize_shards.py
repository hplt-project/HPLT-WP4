# takes in the input directory, output directory, path to the tokenizer, and the max sequence length
# the input directory is the directory containing N sharded jsonl files
# the output directory is the directory where the each file is tokenized

from tokenizers import Tokenizer
import json
import os
import argparse
from smart_open import open
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    return parser.parse_args()


def tokenize(tokenizer, text):
    text = text.rstrip("\n")
    ids = tokenizer.encode(line, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.long)

    return ids


if __name__ == "__main__":
    args = parse_args()

    # make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # remove any existing files in the output directory
    for filename in os.listdir(args.output_dir):
        os.remove(os.path.join(args.output_dir, filename))
    
    # load the tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # iterate through all input files
    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".jsonl") and not filename.endswith(".jsonl.gz"):
            continue

        tokenized_documents = []
        for line in open(os.path.join(args.input_dir, filename), "rt"):
            document = json.loads(line)["text"]
            tokenized_document = tokenize(tokenizer, document)
            tokenized_documents.append(tokenized_document)

        # save the tokenized documents
        output_filename = f"{filename.split('.')[0]}.pt"
        torch.save(tokenized_documents, os.path.join(args.output_dir, output_filename))
