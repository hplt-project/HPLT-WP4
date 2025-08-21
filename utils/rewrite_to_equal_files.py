from glob import glob
import os
import gzip
import torch
import sys
from tqdm import tqdm

def count_total_size(input_dir):
    total_size = 0
    for filename in tqdm(glob(f"{input_dir}/train*")):
        with gzip.GzipFile(filename, 'rb') as f:
            documents = torch.load(f)
        torch.save(documents, 'tmp.pt')
        total_size += os.path.getsize('tmp.pt') # in bytes 

    #total_size /= 1024 * 1024 # in MiB
    return total_size

if __name__ == "__main__":
    input_dir = "/scratch/project_465001890/hplt-2-0-output/eng_Latn-all/eng_Latn/tokenized_shards/"
    total_size = count_total_size(input_dir)
    number_of_shards = 32
    actual_shard_size = total_size / number_of_shards
    print(actual_shard_size / (1024*1024))
    current_documents, current_size = [], 0
    num_scheduled_shards = 0
    shard_dir = os.path.join("/scratch/project_465001890/hplt-2-0-output/eng_Latn-all/eng_Latn/", "tokenized_equal_shards")
    os.makedirs(shard_dir, exist_ok=True)
    filenames = glob(f"{input_dir}/train*")
    filenames.sort()
    for filename in tqdm(filenames):
        print(filename, flush=True)
        with gzip.GzipFile(filename, 'rb') as f:
            documents = torch.load(f)
            for document in documents:
                doc_size = sys.getsizeof(document.storage())
                if current_size + doc_size > actual_shard_size:
                    if num_scheduled_shards > 29:
                        out_name = f"train_{num_scheduled_shards:05d}.pt.gz"
                        print(f"writing {current_size / (1024 * 1024)}M to {out_name}", flush=True)
                        with gzip.GzipFile(f"{shard_dir}/{out_name}", 'wb') as f:
                            torch.save(current_documents, f)
                    current_size = 0
                    current_documents = []
                    num_scheduled_shards += 1
                current_documents.append(document)
                current_size += doc_size
    out_name = f"train_{num_scheduled_shards:05d}.pt.gz"
    print(f"writing {current_size / (1024 * 1024)}M to {out_name}", flush=True)
    with gzip.GzipFile(f"{shard_dir}/{out_name}", 'wb') as f:
        torch.save(current_documents, f)

        