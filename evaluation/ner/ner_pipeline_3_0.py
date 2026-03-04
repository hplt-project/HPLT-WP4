import argparse
from iso639 import Lang
from huggingface_hub import HfApi, get_collection
import subprocess

from constants import LANGS_MAPPING

parser = argparse.ArgumentParser()
parser.add_argument('--collection_slug', default="HPLT/hplt-30-gpt-bert-models")
parser.add_argument('--result_dir', default="/cluster/work/users/mariiaf/ner_results/")
args = parser.parse_args()

collection = get_collection(args.collection_slug)
for item in collection.items:
    print(item)
    if (item.item_type == 'model'):
        try:
            bash_output = subprocess.check_output(f"grep {item.item_id} ner_results.tsv", shell=True)
            print(bash_output.decode("utf-8"), flush=True)
        except subprocess.CalledProcessError:
            model_name = item.item_id.split("/")[1]
            langcode = model_name[-8:-5]
            script = model_name[-4:]
            lg = Lang(langcode)
            pt1 = lg.pt1
            if langcode in {"swh", "ckb", "ast", "ekk", "lvs", "kmr", "cmn", "als"}:
                pt1 = LANGS_MAPPING[langcode+script[0]]
            command = f"sbatch --job-name gpt-bert-NER --output {args.result_dir}/{langcode}-ner-%j.out saga.slurm {item.item_id} wikiann/{pt1} {args.result_dir}/{model_name}"
            bash_output = subprocess.check_output(command, shell=True)
            print(bash_output.decode("utf-8"), flush=True)
