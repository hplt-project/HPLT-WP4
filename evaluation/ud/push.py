import sys
from huggingface_hub import HfApi
from huggingface_hub import create_repo
from huggingface_hub.errors import HfHubHTTPError

try:
    create_repo(sys.argv[1], private=False)
except HfHubHTTPError as e:
    print(e, flush=True)
api = HfApi()

lang = sys.argv[1]
arch = "gpt_bert"
repo_id = f"HPLT/hplt_{arch}_base_3_0_{lang}-UD"
api.upload_folder(folder_path=sys.argv[2], repo_id=repo_id, repo_type="model")
api.add_collection_item(
                collection_slug="HPLT/ud-parsers",
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
)