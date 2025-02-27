import argparse
import json
import logging
import os
import shutil

import torch
from transformers import AutoTokenizer

from constants import LANGS_MAPPING, LANGS_MAPPING_IETF, LANGUAGES

def edit_readme(after_readme, language):
    lang_code = LANGS_MAPPING_IETF[language]
    with open(after_readme, 'r') as file:
        contents = file.read()
    # Replace the word
    new_contents = contents.replace("English", LANGUAGES[lang_code])
    new_contents = new_contents.replace("- en\n", f"- {lang_code}\n")
    new_contents = new_contents.replace("eng-Latn", f"2_0_{LANGS_MAPPING[language]}")
    with open(after_readme, 'w') as file:
        file.write(new_contents)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model_directory",
        type=str,
        default="/scratch/project_465001386/hplt-2-0-output/",
    )
    parser.add_argument(
        "--output_model_directory",
        type=str,
        default="/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/intermediate",
    )
    args = parser.parse_args()
    return args


def convert_hf(input_model_directory, output_model_directory, language, template_dir):
    lang_directory = os.path.join(input_model_directory, language)
    input_model_directory = os.path.join(lang_directory, 'hplt_models', f"bert_base_{language}")
    output_model_directory = os.path.join(
        output_model_directory, f"hplt_bert_base_{LANGS_MAPPING[language]}-intermediate"
    )

    if not os.path.exists(input_model_directory):
        raise ValueError(f"Model directory {input_model_directory} does not exist")

    if not os.path.exists(output_model_directory):
        os.makedirs(output_model_directory, exist_ok=True)


    checkpoints = [
        el.name
        for el in os.scandir(input_model_directory)
        if el.is_file() and el.name.endswith(".bin") and (not '31250' in el.name)
    ]

    for checkpoint in checkpoints:
        logger.info(f"Processing checkpoint {checkpoint}...")
        step_id = "step" + os.path.splitext(checkpoint)[0].split("_")[2]
        step_dir = os.path.join(output_model_directory, step_id)
        _ = shutil.copytree(template_dir, step_dir, dirs_exist_ok=True)

        cur_checkpoint = torch.load(
            os.path.join(input_model_directory, checkpoint),
            map_location=torch.device("cpu"),
        )
        torch.save(cur_checkpoint["model"], os.path.join(step_dir, "pytorch_model.bin"))

        _ = shutil.copy2(
            f"{lang_directory}/tokenizer.json",
            step_dir,
        )

        tokenizer = AutoTokenizer.from_pretrained(step_dir)
        vocabulary_size = tokenizer.vocab_size
        config_dict = json.load(open(os.path.join(step_dir, "config.json")))
        config_dict["vocab_size"] = vocabulary_size
        json.dump(
            config_dict,
            open(os.path.join(step_dir, "config.json"), "w"),
            indent=4,
        )
        # update model card
        after_readme = os.path.join(step_dir, 'README.md')
        edit_readme(after_readme, language)


def main():
    args = parse_args()
    prototype_directory = "huggingface_prototype"
    for language in LANGS_MAPPING.keys():
        print(language)
        convert_hf(
            args.input_model_directory,
            args.output_model_directory,
            language,
            prototype_directory,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    main()
