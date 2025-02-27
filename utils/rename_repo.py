from huggingface_hub import move_repo

from constants import LANGS_MAPPING

def rename_repo():
    for lang in LANGS_MAPPING.values():
        old_name = f"HPLT/hplt_bert_base_{lang}"
        new_name = old_name.replace(lang, f"2_0_{lang}")
        move_repo(from_id=old_name, to_id=new_name)


if __name__ == '__main__':
    rename_repo()