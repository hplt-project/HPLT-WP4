#!/bin/env python3
import os
import sys
from huggingface_hub import HfApi, create_branch
from huggingface_hub.errors import HfHubHTTPError

api = HfApi()

modelpath = sys.argv[1]
branch = sys.argv[2]
print(branch)
print(modelpath)
model = os.path.split(modelpath)[-1].replace("-intermediate", "")
print(model)
# Create branch we want to push to
repo_id = f"HPLT/hplt_bert_base_swh-Latn"
try: # exists_ok is broken in huggingface-hub==0.26.5
    create_branch(repo_id=repo_id, branch=branch.replace('swhLatn_', 'step'))
except HfHubHTTPError:
    print(f"{branch} exists")
folder_path = f"{modelpath}/{branch}"
print(folder_path)
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    revision=branch.replace('swhLatn_', 'step'),
    commit_message=f"Intermediate checkpoint {branch}",
    ignore_patterns="**/logs/*.txt",
)
