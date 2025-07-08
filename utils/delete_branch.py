#!/bin/env python3
import os
import sys
from huggingface_hub import HfApi, delete_branch

api = HfApi()

# Create branch we want to push to
repo_id = f"HPLT/hplt_bert_base_swh-Latn"
for branch in ('swhLatn_3125', 'swhLatn_6250', 'swhLatn_9375'):
    delete_branch(repo_id=repo_id, branch=branch)
