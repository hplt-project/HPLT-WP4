import argparse
import os
from shutil import copyfile

from huggingface_hub import HfApi
from huggingface_hub import create_repo


from constants import LANGS_MAPPING, LANGS_MAPPING_IETF, LANGUAGES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', default='engL,belC')
    parser.add_argument('--hf_models_path', default='/scratch/project_465001890/hplt-2-0-output/hplt_hf_models/')
    return parser.parse_args()


def rename_folder(before, after):
    print(f"Renaming {before}")
    os.rename(before, after)

if __name__ == '__main__':
    args = parse_args()
    args.languages = [lang for lang in args.languages.split(',') if lang]
    for language in args.languages:
        before = os.path.join(args.hf_models_path, language)
        after = before.replace(language, LANGS_MAPPING[language])
        if os.path.exists(before):
            rename_folder(before, after)
        else:
            before += '_31250' # mess with namings after saving additional checkpoints
            if os.path.exists(before):
                rename_folder(before, after)

        print(after)
        if not os.path.exists(after):
            after += '_31250'
        after_readme = f"{after}/README.md"

        if not os.path.exists(after_readme):
            lang_code = LANGS_MAPPING_IETF[language]
            
            
            spacial_tokens_map = os.path.join(after, 'spacial_tokens_map.json')
            if os.path.exists(spacial_tokens_map):
                os.rename(spacial_tokens_map, os.path.join(after, 'special_tokens_map.json'))
            
            copyfile('../encoder-only/huggingface_prototype/README.md', after_readme)
            with open(after_readme, 'r') as file:
                contents = file.read()
            # Replace the word
            new_contents = contents.replace("English", LANGUAGES[lang_code])
            new_contents = new_contents.replace("- en\n", f"- {lang_code}\n")
            new_contents = new_contents.replace("eng-Latn", LANGS_MAPPING[language])
            with open(after_readme, 'w') as file:
                file.write(new_contents)

        repo_id = f"HPLT/hplt_bert_base_{LANGS_MAPPING[language]}"
        create_repo(repo_id)
        api = HfApi()

        api.upload_folder(folder_path=after, repo_id=repo_id, repo_type="model")
