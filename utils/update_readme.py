import argparse

from huggingface_hub import HfApi, hf_hub_download, get_collection

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_slug', default="HPLT/hplt-30-t5-models")
    return parser.parse_args()

if __name__ == '__main__':
    api = HfApi()
    args = parse_args()
    collection = get_collection(args.collection_slug)
    for item in collection.items:
        print(item.item_id)
        readme_path_in_repo = "README.md"
        main_readme_path = hf_hub_download(repo_id=item.item_id, filename=readme_path_in_repo)
        with open('before.txt', 'r') as f:
            before = f.read()
        with open('after.txt', 'r') as f:
            after = f.read()
        with open(main_readme_path, 'r') as file:
            contents = file.read()
        contents = contents.replace(before, after)
        with open(main_readme_path, 'w') as file:
            file.write(contents)
        refs = api.list_repo_refs(item.item_id)
        for branch in refs.branches:
            api.upload_file(
                path_or_fileobj=main_readme_path,
                path_in_repo=readme_path_in_repo,
                repo_id=item.item_id,
                repo_type="model",
                commit_message=f"Updating README",
                revision=branch.name,
            )
