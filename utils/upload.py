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

api.upload_folder(folder_path=sys.argv[2], repo_id=sys.argv[1], repo_type="model")

