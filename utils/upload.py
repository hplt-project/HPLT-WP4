#!/bin/env python3

import sys
from huggingface_hub import HfApi
from huggingface_hub import create_repo

create_repo(sys.argv[1])
api = HfApi()

api.upload_folder(folder_path=sys.argv[2], repo_id=sys.argv[1], repo_type="model")

