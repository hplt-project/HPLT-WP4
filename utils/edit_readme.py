#! /bin/env/python3
import argparse
from glob import glob
import os

from iso639 import Lang
from huggingface_hub import HfApi

from constants import LANGS_MAPPING_IETF

api = HfApi()
parser = argparse.ArgumentParser()
parser.add_argument('--path', default="/scratch/project_465002259/hplt-3-0-output/hf_models/")
args = parser.parse_args()

for folder in glob(args.path + '*'):
    el = os.path.split(folder)[-1]
    
    if os.path.isdir(folder) and '_' in el:
        print(el)
        langcode, script, _ = el.split("_") # als, Latn
        lg = Lang(langcode)

        with open("../encoder-decoder/huggingface_prototype/README.md", 'r') as file:
            contents = file.read()
        # Replace the word
        name = lg.name
        if lg.other_names():
            try:
                name += f" ({', '.join(lg.other_names())})"
            except TypeError:
                print(lg.other_names)
        new_contents = contents.replace("English", name)
        lg_pt1 = lg.pt1
        if langcode in {"swh", "ckb", "ast", "ekk", "lvs", "kmr", "cmn", "als"}:
            lg_pt1 = LANGS_MAPPING_IETF[langcode+script[0]]
        new_contents = new_contents.replace("- en\n", f"- {lg_pt1}\n")
        new_contents = new_contents.replace("- eng", f"- {langcode}")
        new_contents = new_contents.replace("_eng_Latn", f"_{langcode}_{script}")
        out = os.path.join(folder, "README.md")
        with open(out, 'w') as file:
            file.write(new_contents)
        if "31250" in el:  
            api.upload_file(
                path_or_fileobj=out,
                path_in_repo="README.md",
                repo_id=f"HPLT/hplt_t5_base_3_0_{langcode}_{script}",
                repo_type="model",
                commit_message=f"Updating README",
            )