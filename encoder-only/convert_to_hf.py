import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer

STEP_PATTERN = re.compile(r"\d+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_directory', type=str, default='/scratch/project_465001890/hplt-2-0-output/')
    parser.add_argument('--output_model_directory', type=str, default='~/hplt_hf_models')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--all_checkpoints', action='store_true')
    args = parser.parse_args()
    return args


def convert_to_hf(
        input_model_directory,
        output_model_directory,
        language,
        all_checkpoints,
):
    checkpointing_steps = [31250]
    checkpoints_directory = os.path.join(
        input_model_directory, f"{language}/hplt_models/bert_base_{language}",
    )
    if all_checkpoints:
        print(f"Files in the checkpoints_directory: {os.listdir(checkpoints_directory)}")
        checkpointing_steps = [
            int(re.search(STEP_PATTERN, bin_name).group(0)) for bin_name in os.listdir(checkpoints_directory)
        ]
        print(f"Saving steps {checkpointing_steps}")
    for step in checkpointing_steps:
        step_output_model_directory = os.path.join(output_model_directory, language+f'_{step}')
        checkpoint_path = os.path.join(
            checkpoints_directory, f"model_step_{step}.bin",
        )
        if not os.path.exists(checkpoints_directory):
            raise ValueError(f"Model directory {checkpoints_directory} does not exist")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Model file {checkpoint_path} does not exist")

        if not os.path.exists(step_output_model_directory):
            os.makedirs(step_output_model_directory)

        prototype_directory = "huggingface_prototype"
        os.system(f"cp {prototype_directory}/* {step_output_model_directory}")

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        torch.save(checkpoint['model'], os.path.join(step_output_model_directory, "pytorch_model.bin"))

        os.system(f"cp {input_model_directory}/{language}/tokenizer.json {step_output_model_directory}")

        tokenizer = AutoTokenizer.from_pretrained(step_output_model_directory)
        vocabulary_size = tokenizer.vocab_size
        config_dict = json.load(open(os.path.join(step_output_model_directory, "config.json")))
        config_dict["vocab_size"] = vocabulary_size
        json.dump(
            config_dict,
            open(os.path.join(step_output_model_directory, "config.json"), "w"),
            indent=4,
        )


def main():
    args = parse_args()
    convert_to_hf(
        args.input_model_directory,
        args.output_model_directory,
        args.language,
        args.all_checkpoints,
    )


if __name__ == "__main__":
    main()