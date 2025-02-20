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
        default='/scratch/project_465001386/hplt-2-0-output/fraL/tokenized_shards,'
                '/scratch/project_465001386/hplt-2-0-output/cesL/tokenized_shards,'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print(args)
    for problematic_folder in args.problematic_folders.split(','):
        current_number = 0
        number_of_train_docs = 0
        print(problematic_folder)
        paths = glob(f"{problematic_folder}/*.pt.gz")
        paths.sort()
        for file_path in tqdm(paths):
            if os.stat(file_path).st_size == 0:
                os.remove(file_path)
                if 'validation' in file_path:
                    print("Validation file does not exist!")
            else:
                with gzip.GzipFile(file_path, 'rb') as f:
                    try:
                        documents = torch.load(f)
                    except EOFError:
                        print(f"Broken file {file_path}, removing")
                        os.remove(file_path)
                        continue

                if 'train' in file_path:
                    print(current_number)
                    number_of_train_docs += len(documents)
                    new_path = os.path.join(problematic_folder, f'train_{current_number:05d}.pt.gz')
                    print(file_path)
                    print(new_path)
                    os.rename(file_path, new_path)
                    current_number += 1

        print(f"{number_of_train_docs} training documents in {problematic_folder}")
