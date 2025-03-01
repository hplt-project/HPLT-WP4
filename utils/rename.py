"""
Make 2.0 data path names compatible with the code for 1.0
"""
import argparse
import os
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='/scratch/project_465001386/hplt-2-0-full')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    for dir_path, dir_name in zip(glob(f"{args.dir}/*"), os.listdir(args.dir)):
        if os.path.isdir(dir_path):
            print(f"Renaming {dir_path}")
            
            try:
                lang = dir_name[:3] + dir_name[4]
            except IndexError:
                continue # dir already processed
            
            new_dir_path = f"{args.dir}/{lang}"
            if not os.path.exists(new_dir_path):
                try:
                    os.rename(dir_path, new_dir_path)
                except OSError as e: # ongoing download
                    print(e)
                    continue
                for fn in glob(f"{new_dir_path}/*.jsonl.zst"):
                    new_fn = os.path.join(new_dir_path, f"{lang}_{os.path.split(fn)[-1].split('_')[-1]}")
                    print(new_fn)
                    os.rename(fn, new_fn)
