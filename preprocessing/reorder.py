import argparse
from glob import glob
import gzip
import os

import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problematic_folders',
        default='/scratch/project_465001386/hplt-2-0-output/fraL/tokenized_shards/,'
                '/scratch/project_465001386/hplt-2-0-output/cesL/tokenized_shards/,'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    current_number = 0
    number_of_train_docs = 0
    print(args)
    for problematic_folder in args.problematic_folders.split(','):
        print(problematic_folder)
        for file_path in tqdm(glob(f"{problematic_folder}/*.pt.gz")):
            if os.stat(file_path).st_size == 0:
                os.remove(file_path)
                if 'validation' in file_path:
                    print("Validation file does not exist!")
            else:
                if 'train' in file_path:
                    new_path = os.path.join(problematic_folder, f'train_{current_number:05d}.pt.gz')
                    os.rename(file_path, new_path)
                    current_number += 1
                    with gzip.GzipFile(new_path, 'rb') as f:
                        documents = torch.load(f)
                        number_of_train_docs += len(documents)
        print(f"{number_of_train_docs} training documents in {problematic_folder}")