#!/bin/env python3

import argparse

from huggingface_hub import HfApi

from constants import LANGUAGES, LANGS_MAPPING


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='2_0', choices=('3_0', '2_0', '1_2'))
    parser.add_argument('--collection_slug', default="HPLT/hplt-20-bert-models-67ba52ae96b1fb8aae673493")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.version == '1_2':
        languages = LANGUAGES
    else:
        languages = tuple(LANGS_MAPPING.values())

    api = HfApi()
    arch = "bert"
    for lang in languages:
        if args.version == '3_0':
            lang = lang.replace("-", "_")
            arch = "t5"
        print(lang)
        try:
            api.add_collection_item(
                collection_slug=args.collection_slug,
                item_id=f"HPLT/hplt_{arch}_base_{args.version}_{lang}", 
                item_type="model",
                exists_ok=True,
                )
        except:
            print("doesn't exist")
