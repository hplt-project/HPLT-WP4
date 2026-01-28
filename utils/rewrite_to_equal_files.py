import argparse
from glob import glob
import os
import gzip
import torch
import sys
from tqdm import tqdm
import gc
from shutil import rmtree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/scratch/project_465002259/hplt-3-0-t5/")
    parser.add_argument('--folder', default='tokenized_neq_shards/')
    parser.add_argument('--out_dir', default='tokenized_shards')
    parser.add_argument('--lang', default="deu_Latn")
    parser.add_argument('--shard_size', type=float, default=0.0)
    parser.add_argument('--num_files', type=int, default=0)
    parser.add_argument('--total', type=float, default=2349.8613547086716*1024*1024*30)
    return parser.parse_args()


def count_total_size(input_dir, lang):
    total_size = 0
    for filename in tqdm(glob(f"{input_dir}/train*")):
        try:
            with gzip.GzipFile(filename, 'rb') as f:
                documents = torch.load(f)
        except EOFError:
            print(f"{filename} is empty")
            continue
        torch.save(documents, f'tmp-{lang}.pt')
        total_size += os.path.getsize(f'tmp-{lang}.pt') # in bytes 
        del documents
        gc.collect()
    return total_size


if __name__ == "__main__":
    args = parse_args()
    input_dir = os.path.join(args.input_dir, args.lang, args.folder)
    if args.num_files:
        args.out_dir = args.out_dir + f"_{args.num_files}"
    if not args.total:
        args.num_files += 1 # for languages smaller than German
    print(f"{args}", flush=True)
    number_of_shards = 32
    if (not args.shard_size) and (not args.num_files):
        total_size = count_total_size(input_dir, args.lang)
        actual_shard_size = total_size / number_of_shards
    elif args.shard_size:
        actual_shard_size = args.shard_size * (1024*1024)
    else:
        if args.total:
            actual_shard_size = args.total / args.num_files
        else:
            total_size = count_total_size(input_dir, args.lang)
            actual_shard_size = total_size / args.num_files
    print(actual_shard_size / (1024*1024), flush=True)
    
    current_documents, current_size = [], 0
    num_scheduled_shards = 0
    shard_dir = os.path.join(args.input_dir, args.lang, args.out_dir)
    if os.path.isdir(shard_dir):
        rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    filenames = glob(f"{input_dir}/train*")
    filenames.sort()
    for i, filename in enumerate(tqdm(filenames)):
        print(filename, flush=True)
        if len(os.listdir(shard_dir)) >= args.num_files:
            break
        with gzip.GzipFile(filename, 'rb') as f:
            try:
                documents = torch.load(f)
            except EOFError:
                continue
            for document in documents:
                doc_size = sys.getsizeof(document.storage())
                if current_size + doc_size > actual_shard_size:
                    if not args.num_files:
                        out_name = f"train_{num_scheduled_shards:05d}.pt.gz"
                    else:
                        out_name = f"train-{num_scheduled_shards:03d}.pt.gz"
                    print(f"writing {current_size / (1024 * 1024)}M to {out_name}", flush=True)
                    with gzip.GzipFile(f"{shard_dir}/{out_name}", 'wb') as f:
                        torch.save(current_documents, f)
                    current_size = 0
                    current_documents = []
                    num_scheduled_shards += 1
                    if len(os.listdir(shard_dir)) >= args.num_files:
                        break
                current_documents.append(document)
                current_size += doc_size
    # we don't write the last file, because it is likely to contain too few documents
    # if len(os.listdir(shard_dir)) < args.num_files:
    #     out_name = f"train-{num_scheduled_shards:03d}.pt.gz"
    #     print(f"writing {current_size / (1024 * 1024)}M to {out_name}", flush=True)
    #     with gzip.GzipFile(f"{shard_dir}/{out_name}", 'wb') as f:
    #         torch.save(current_documents, f)

        