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
        lang = dir_name[:3] + dir_name[5]
        new_dir_path = f"{args.dir}/{lang}"
        os.rename(dir_path, new_dir_path)
        for fn in glob(f"{new_dir_path}/*.jsonl.zst"):
            new_fn = f"{lang}_{os.path.split(fn)[-1]}"
            os.rename(fn, new_fn)
