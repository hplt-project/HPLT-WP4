#!/bin/env python3
import argparse
from glob import glob
import os

from huggingface_hub import HfApi, create_branch
from huggingface_hub.errors import HfHubHTTPError

api = HfApi()

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="/scratch/project_465002259/hplt-3-0-output/hf_models/")
args = parser.parse_args()

for folder in glob(args.path + 'mlt_*'):
    el = os.path.split(folder)[-1]
    
    if os.path.isdir(folder) and ('_' in el) and ('31250' not in el):
        langcode, script, _ = el.split("_") # als, Latn
        print(folder)
        # Create branch we want to push to
        repo_id = f"HPLT/hplt_t5_base_3_0_{langcode}_{script}"
        branch = el.replace(f'{langcode}_{script}_', 'step')
        try: # exists_ok is broken in huggingface-hub==0.26.5
            create_branch(repo_id=repo_id, branch=branch)
        except HfHubHTTPError:
            print(f"{el} exists")
        api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message=f"Intermediate checkpoint {branch}",
            ignore_patterns="**/logs/*.txt",
        )
