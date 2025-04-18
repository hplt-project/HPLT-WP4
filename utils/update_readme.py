from huggingface_hub import HfApi, hf_hub_download
from constants import LANGS_MAPPING


if __name__ == '__main__':
    api = HfApi()
    for short, long in LANGS_MAPPING.items():
        print(long)
        problem_repo = f'HPLT/hplt_bert_base_2_0_{long}'
        readme_path_in_repo = "README.md"
        main_readme_path = hf_hub_download(repo_id=problem_repo, filename=readme_path_in_repo)
        with open('before.txt', 'r') as f:
            before = f.read()
        with open('after.txt', 'r') as f:
            after = f.read()
        with open(main_readme_path, 'r') as file:
            contents = file.read()
        contents = contents.replace(before, after)
        with open(main_readme_path, 'w') as file:
            file.write(contents)
        refs = api.list_repo_refs(problem_repo)
        for branch in refs.branches:
            api.upload_file(
                path_or_fileobj=main_readme_path,
                path_in_repo=readme_path_in_repo,
                repo_id=problem_repo,
                repo_type="model",
                commit_message=f"Updating README",
                revision=branch.name,
            )
