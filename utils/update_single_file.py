import argparse
from huggingface_hub import HfApi


def update_repo(args):
    api = HfApi()
    refs = api.list_repo_refs(args.repo)
    api.upload_file(
        path_or_fileobj=args.main_path,
        path_in_repo=args.path_in_repo,
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.msg,
    )
    for branch in refs.branches:
        api.upload_file(
            path_or_fileobj=args.main_path,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo,
            repo_type="model",
            commit_message=args.msg,
            revision=branch.name,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('main_path')
    parser.add_argument('--repo', default="HPLT/hplt_gpt_bert_base_3_0_deu_Latn")
    parser.add_argument('--path_in_repo', default="config.json")
    parser.add_argument('--msg', default="Prettify config")
    args = parser.parse_args()
    update_repo(args)