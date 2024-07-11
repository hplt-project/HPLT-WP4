#!/bin/env python3

import sys
from huggingface_hub import HfApi

api = HfApi()

modelpath = sys.argv[1]
file2upload = sys.argv[2]
model = modelpath[:-1]

api.upload_file(
    path_or_fileobj=f"{modelpath}{file2upload}",
    path_in_repo=file2upload,
    repo_id=f"HPLT/{model}",
    repo_type="model",
    commit_message=f"Updating {file2upload}",
)
