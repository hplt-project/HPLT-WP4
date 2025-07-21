import argparse

from huggingface_hub import get_collection, HfApi, delete_file, HfFileSystem

parser = argparse.ArgumentParser()
parser.add_argument('--to_remove', default='model.safetensors')
parser.add_argument('--collection_slug', default='HPLT/hplt-12-bert-models-6625a8f3e0f8ed1c9a4fa96d')
args = parser.parse_args()

api = HfApi()
fs = HfFileSystem()
collection = get_collection(args.collection_slug)
for item in collection.items:
    print(item.item_id)
    files = [file["name"] for file in fs.ls(item.item_id)]
    if '/'.join((item.item_id, args.to_remove)) in files:
        delete_file(path_in_repo=args.to_remove, repo_id=item.item_id, commit_message='rm broken safetensors')
        print('Deleted')