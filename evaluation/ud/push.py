import sys
from huggingface_hub import HfApi
from huggingface_hub import create_repo

lang = sys.argv[1]
arch = "gpt_bert"
repo_id = f"HPLT/hplt_{arch}_base_4_0_{lang}-UD"
try:
    create_repo(repo_id, private=False)
except Exception as e: # exact error depends on hf version
    print(e, flush=True)
api = HfApi()


api.upload_folder(folder_path=sys.argv[2], repo_id=repo_id, repo_type="model")
api.add_collection_item(
                collection_slug="HPLT/ud-parsers",
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
)