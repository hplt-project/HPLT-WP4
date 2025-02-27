#!/bin/env python3
import os
import sys
from huggingface_hub import HfApi, create_branch
from huggingface_hub.errors import HfHubHTTPError

api = HfApi()

modelpath = sys.argv[1]
branch = sys.argv[2]
print(modelpath)
model = os.path.split(modelpath)[-1].replace("-intermediate", "")
print(model)
# Create branch we want to push to
try: # exists_ok is broken in 
    create_branch(repo_id=f"HPLT/{model}", branch=branch)
except HfHubHTTPError:
    print(f"{branch} exists")

api.upload_folder(
    folder_path=f"{modelpath}/{branch}",
    repo_id=f"HPLT/{model}",
    repo_type="model",
    revision=branch,
    commit_message=f"Intermediate checkpoint {branch}",
    ignore_patterns="**/logs/*.txt",
)
