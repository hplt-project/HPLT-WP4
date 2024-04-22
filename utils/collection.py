#!/bin/env python3

import sys
from huggingface_hub import HfApi
from huggingface_hub import create_repo
import json
import os

with open("ISO-639-1-language.json") as f:
    a = json.load(f)

languages = {}

for el in a:
    code = el["code"]
    name = el["name"]
    languages[code] = name

api = HfApi()

for lang in languages:
    print(lang)
    try:
        api.add_collection_item(collection_slug="HPLT/hplt-bert-models-6625a8f3e0f8ed1c9a4fa96d", item_id=f"HPLT/hplt_bert_base_{lang}", item_type="model", exists_ok=True)
    except:
        print("doesn't exist")
