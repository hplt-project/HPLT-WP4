import argparse
from glob import glob
import json
import os
from shutil import copyfile

from constants import LANGS_MAPPING, LANGS_MAPPING_IETF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='engL')
    parser.add_argument('--hf_models_path', default='/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    before = os.path.join(args.hf_models_path, args.language)
    lang_code = LANGS_MAPPING_IETF[args.language]
    with open("ISO-639-1-language.json") as f:
        lang_code2word = json.load(f)
    languages = {}
    for line in lang_code2word:
        code = line["code"]
        name = line["name"]
        languages[code] = name
    
    print(before)
    after = before.replace(args.language, LANGS_MAPPING[args.language])
    print(after)
    os.rename(before, after)
    os.rename(os.path.join(after, 'spacial_tokens_map.json'), os.path.join(after, 'special_tokens_map.json'))
    after_readme = f"{after}/README.md"
    copyfile('../encoder-only/huggingface_prototype/README.md', after_readme)
    with open(after_readme, 'r') as file:
        contents = file.read()
    # Replace the word
    new_contents = contents.replace("English", languages[lang_code])
    new_contents = new_contents.replace("- en\n", f"- {lang_code}\n")
    new_contents = new_contents.replace("eng-Latn", LANGS_MAPPING[args.language])
    with open(after_readme, 'w') as file:
        file.write(new_contents)
