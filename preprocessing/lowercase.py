import argparse
from glob import glob
import gzip
import json
import os
import random

from smart_open import open

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/cluster/work/projects/nn9851k/mariiaf/hplt/")
    parser.add_argument("--langs", default="dan_Latn,nno_Latn,nob_Latn")
    parser.add_argument("--out", default="/cluster/work/projects/nn9851k/mariiaf/dan_nob_nno/")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sampled = []
    for lang in args.langs.split(","):
        data_files = glob(os.path.join(args.data, lang, "text_shards/train*"))
        sampled += random.sample(data_files, 8)
    random.shuffle(sampled)
    assert len(sampled) == 3 * 8
    count = 0
    for input_file in sampled:
        print(input_file, flush=True)
        with gzip.open(os.path.join(args.out, f"train_{count:05d}.jsonl.gz"), "wt") as out_file:
            for i, document in enumerate(open(input_file, 'rt')):
                text = json.loads(document)
                if i == 0:
                    print(f"Before: {text}", flush=True)
                    print("-----------------------------------------------------------", flush=True)
                text = text.lower()
                if i == 0:
                    print(f"After: {text}", flush=True)
                out_file.write(json.dumps(text) + "\n")
        count += 1
        
