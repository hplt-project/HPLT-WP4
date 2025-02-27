from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from constants import LANGS_MAPPING

def rename_repo():
    api = HfApi()
    for lang in LANGS_MAPPING.values():
        print(lang)
        problem_repo = f'HPLT/hplt_bert_base_{lang}'
        refs = api.list_repo_refs(problem_repo)
        readme_path_in_repo = "README.md"
        main_readme_path = hf_hub_download(repo_id=problem_repo, filename=readme_path_in_repo)
        with open(main_readme_path, 'r') as file:
            contents = file.read()
        contents = contents.replace(f"base_2_0_2_0_2_0_{lang}", f"base_2_0_{lang}")
        contents = contents.replace(f"base_2_0_2_0_{lang}", f"base_2_0_{lang}")
        contents = contents.replace(f"base_{lang}", f"base_2_0_{lang}")
        with open(main_readme_path, 'w') as file:
            file.write(contents)
        api.upload_file(
            path_or_fileobj=main_readme_path,
            path_in_repo=readme_path_in_repo,
            repo_id=problem_repo,
            repo_type="model",
            commit_message=f"Updating README with 2.0 in name",
        )
        for branch in refs.branches:
            try:
                api.delete_file('spacial_tokens_map.json', problem_repo, revision=branch.name)
            except EntryNotFoundError:
                print(f"From {branch.name} spacial_tokens_map already removed")


if __name__ == '__main__':
    rename_repo()