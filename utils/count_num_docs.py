import zstandard as zstd
import os
import io
import gzip
import torch
from tqdm import tqdm
from glob import glob
import argparse
import random
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/scratch/project_465002259/hplt-3-0-t5/deu_Latn/tokenized_equal_shards/")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    return args

def zst():
    """
    Randomly sample 100 best documents from a language
    Count number of documents in *.zst
    """
    folder="/appl/local/openeurollm/training/catalogue/hplt/3.0/sorted/nno_Latn"
    fn="10_1.jsonl.zst"
    dctx = zstd.ZstdDecompressor()
    with open(os.path.join(folder, fn), "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for num_samples, line in enumerate(text_stream):
                continue
    print(f"Number of documents in {fn}: {num_samples}")
    k=100
    indices_to_pick = set(random.sample([i for i in range(num_samples + 1)], k=k))
    n_processed_train_documents = 0
    with open(os.path.join(folder, fn), "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for i, line in enumerate(tqdm(text_stream)):        
                if n_processed_train_documents != k:
                    if i in indices_to_pick:
                        line = json.loads(line)
                        document = line["text"].strip()
                        if len(document) == 0:
                            indices_to_pick.add(random.choice([j for j in range(i + 1, num_samples + 1) if j not in indices_to_pick]))
                            continue

                        print(f"\ndocument{i}: {document}\n\n", flush=True)
                        n_processed_train_documents += 1
                        indices_to_pick.remove(i)

def count_tokenized(num_doc, num_toks, args):
    """
    Count number of documents in a tokenized shard (*.pt.gz)
    """
    for input_file in glob(f"{args.path}/{args.split}.pt.gz"):
        print(input_file)
        with gzip.GzipFile(input_file, 'rb') as f:
            documents = torch.load(f)
            len_file = len(documents)
            print(len_file)
            len_toks = sum([len(x) for x in documents])
            print(len_toks)
            num_toks +=len_toks
            num_doc += len_file
    return num_doc, num_toks

def txt_gzip(args):
    """
    Count number of documents in a not yet tokenized shard (*.jsonl.gz)
    """
    num_doc = 0
    num_toks = 0
    for input_file in glob(os.path.join(args.path, f"{args.split}*")):
        print(input_file, flush=True)
        with gzip.GzipFile(input_file, 'rb') as f:
            for _ in f:
                num_doc += 1
    print(f"Number of documents : {num_doc}")
    print(f"Number of tokens : {num_toks}")

if __name__ == "__main__":
    args = parse_args()
    zst()
