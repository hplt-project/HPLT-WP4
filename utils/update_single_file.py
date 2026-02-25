import argparse
from huggingface_hub import HfApi, get_collection


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
    parser.add_argument('--collection_slug', default="")
    parser.add_argument('--path_in_repo', default="config.json")
    parser.add_argument('--msg', default="Prettify config")
    args = parser.parse_args()

    if args.collection_slug:
        collection = get_collection(args.collection_slug)
        for item in collection.items:
            print(item)
            if item.item_type == 'model':
                args.repo = item.item_id
                update_repo(args)
    else:
        update_repo(args)
