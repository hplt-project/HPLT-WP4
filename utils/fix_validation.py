import os
import gzip
from glob import glob
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/scratch/project_465002259/hplt-3-0-t5/rus_Cyrl/text_shards/")
    args = parser.parse_args()
    return args


def fix_text_shards(args):
    input_file = os.path.join(args.path, "validation.jsonl.gz")
    num_doc = 0
    if os.path.exists(input_file):
        print(input_file, flush=True)
        with gzip.GzipFile(input_file, 'rb') as f:
            for _ in f:
                num_doc += 1
        print(f"Number of documents : {num_doc}")
    with gzip.GzipFile(input_file, 'ab') as f_val:
        for train in glob(os.path.join(args.path, "train*")):
            print(train, flush=True)
            tmp = os.path.join(args.path, "tmp.jsonl.gz")
            with gzip.GzipFile(tmp, 'wb') as f:
                with gzip.GzipFile(train, 'rb') as f_train:
                    for line in f_train:
                        if num_doc < 10000:
                            f_val.write(line)
                            num_doc += 1
                        else:
                            f.write(line)
            os.rename(tmp, train)
            if num_doc > 9999:
                break

def fix_pt(args):
    file_to_split = os.path.join(args.path, "train_00036.pt.gz")
    with gzip.GzipFile(file_to_split, 'rb') as f:
        documents = torch.load(f)
    val_documents = documents[:10_000]
    documents = documents[10_000:]
    with gzip.GzipFile(os.path.join(args.path, "validation.pt.gz"), 'wb') as f:
        torch.save(val_documents, f)
    with gzip.GzipFile(os.path.join(args.path, "train_00037.pt.gz"), 'wb') as f:
        torch.save(documents, f)

if __name__ == "__main__":
    args = parse_args()
    fix_text_shards(args)
