from tokenizers import decoders, Regex, Tokenizer
from huggingface_hub import HfApi, hf_hub_download
from constants import LANGS_MAPPING

if __name__ == '__main__':
    api = HfApi()
    for short, long in LANGS_MAPPING.items():
        print(long)
        tok_path = f"/scratch/project_465001386/hplt-2-0-output/{short}/tokenizer.json"
        problem_repo = f'HPLT/hplt_bert_base_2_0_{long}'
        readme_path_in_repo = "README.md"
        main_readme_path = hf_hub_download(repo_id=problem_repo, filename=readme_path_in_repo)
        bad = 'HPLT Bert'
        with open(main_readme_path, 'r') as file:
            contents = file.read()
        replace_readme = bad in contents
        if replace_readme:
            contents = contents.replace(bad, 'HPLT v2.0 BERT').replace(
                'A monolingual LTG-BERT model is trained for some languages in the [HPLT 2.0 data release](https://hplt-project.org/datasets/v2.0)',
                'We present monolingual LTG-BERT models for more than 50 languages out of 191 total in the [HPLT v2.0 dataset](https://hplt-project.org/datasets/v2.0)',
                )
            with open(main_readme_path, 'w') as file:
                file.write(contents)            

        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.decoder = decoders.Sequence([
                decoders.ByteLevel(add_prefix_space=False, use_regex=False),
                decoders.Replace('âĸģ', ' '),
                decoders.Replace(Regex("█ "), "\n"),
                decoders.Replace(Regex("█"), "\n"),
                decoders.Replace('▁', ' '),
                decoders.Strip(' ', 1, 0),
            ])
        enc = tokenizer.encode('Meow meow')
        saved = f"/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/{long}/tokenizer.json"
        try:
            tokenizer.save(saved)
        except Exception:
            saved = f"/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/{long}_31250/tokenizer.json"
            tokenizer.save(saved)
        refs = api.list_repo_refs(problem_repo)
        for branch in refs.branches:
            api.upload_file(
                    path_or_fileobj=saved,
                    path_in_repo="tokenizer.json",
                    repo_id=problem_repo,
                    repo_type="model",
                    commit_message=f"Fix tokenizer decoders",
                    revision=branch.name,
                )
            if replace_readme:
                api.upload_file(
                    path_or_fileobj=main_readme_path,
                    path_in_repo=readme_path_in_repo,
                    repo_id=problem_repo,
                    repo_type="model",
                    commit_message=f"Updating README with 2.0 in name",
                    revision=branch.name,
                )
