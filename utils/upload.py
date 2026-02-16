#!/bin/env python3

import sys
from huggingface_hub import HfApi
from huggingface_hub import create_repo
from huggingface_hub.errors import HfHubHTTPError

try:
    create_repo(sys.argv[1], private=False)
except HfHubHTTPError as e:
    print(e, flush=True)
api = HfApi()

lang = sys.argv[3]
api.upload_folder(folder_path=sys.argv[2], repo_id=sys.argv[1], repo_type="model")
arch="gpt_bert"
api.add_collection_item(
                collection_slug="HPLT/hplt-30-gpt-bert-models",
                item_id=f"HPLT/hplt_{arch}_base_3_0_{lang}", 
                item_type="model",
                exists_ok=True,
                )
