import os
import subprocess
from huggingface_hub import HfApi
from constants import LANGS_MAPPING

ALL_LANGS = "/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/intermediate"

def fix():
    for lang in LANGS_MAPPING.values():
        api = HfApi()
        print(lang)
        problem_repo = f'HPLT/hplt_bert_base_{lang}'
        refs = api.list_repo_refs(problem_repo)
        for branch in refs.branches:
            if 'step' in branch.name:
                commits = api.list_repo_commits(problem_repo, revision=branch.target_commit)
                last_commit = commits[0]
                if not 'Intermediate' in last_commit.title:
                    print(branch.name)
                    ckpt_path = os.path.join(ALL_LANGS, f"hplt_bert_base_{lang}-intermediate")
                    subprocess.run(["python3", "upload_branch.py", ckpt_path, branch.name])


if __name__ == '__main__':
    fix()