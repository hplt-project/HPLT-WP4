#!/bin/env python3

import argparse

from huggingface_hub import HfApi

from constants import LANGUAGES, LANGS_MAPPING


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='2_0', choices=('2_0', '1_2'))
    parser.add_argument('--collection_slug', default="HPLT/hplt-20-bert-models-67ba52ae96b1fb8aae673493")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.version == '1_2':
        languages = LANGUAGES
    elif args.version == '2_0':
        languages = tuple(LANGS_MAPPING.values())

    api = HfApi()

    for lang in languages:
        print(lang)
        try:
            api.add_collection_item(
                collection_slug=args.collection_slug,
                item_id=f"HPLT/hplt_bert_base_{lang}", 
                item_type="model",
                exists_ok=True,
                )
        except:
            print("doesn't exist")
