from huggingface_hub import HfApi

from constants import LANGS_MAPPING


def update_repo():
    api = HfApi()
    for lang in LANGS_MAPPING.values():
        print(lang)
        problem_repo = f'HPLT/hplt_bert_base_2_0_{lang}'
        refs = api.list_repo_refs(problem_repo)
        main_path = '../encoder-only/huggingface_prototype/modeling_ltgbert.py'
        api.upload_file(
            path_or_fileobj=main_path,
            path_in_repo='modeling_ltgbert.py',
            repo_id=problem_repo,
            repo_type="model",
            commit_message=f"Fix AttributeError in _init_weights for LayerNorm",
        )
        for branch in refs.branches:
            api.upload_file(
                path_or_fileobj=main_path,
                path_in_repo='modeling_ltgbert.py',
                repo_id=problem_repo,
                repo_type="model",
                commit_message=f"Fix AttributeError in _init_weights for LayerNorm",
                revision=branch.name,
            )


if __name__ == '__main__':
    update_repo()