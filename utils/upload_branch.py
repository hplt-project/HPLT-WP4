#!/bin/env python3

import sys
from huggingface_hub import HfApi, create_branch

api = HfApi()

modelpath = sys.argv[1]
branch = sys.argv[2]

model = modelpath.replace("-intermediate/", "")

# Create branch we want to push to
create_branch(repo_id=f"HPLT/{model}", branch=branch)

api.upload_folder(
    folder_path=f"{modelpath}{branch}",
    repo_id=f"HPLT/{model}",
    repo_type="model",
    revision=branch,
    commit_message=f"Intermediate checkpoint {branch}",
    ignore_patterns="**/logs/*.txt",
)
