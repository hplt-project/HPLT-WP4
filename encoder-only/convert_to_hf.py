import os
import torch
import argparse
import json

from transformers import AutoTokenizer
from train import FREQUENT_CHECKPOINTING, FREQUENT_CHECKPOINTING_STEPS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_directory', type=str, default='/scratch/project_465001386/hplt-2-0-output/')
    parser.add_argument('--output_model_directory', type=str, default='~/hplt_hf_models')
    parser.add_argument('--language', type=str, default='en')
    args = parser.parse_args()
    return args


def convert_to_hf(input_model_directory, output_model_directory, language):
    checkpointing_steps = [31250]
    if language in FREQUENT_CHECKPOINTING:
        checkpointing_steps += list(FREQUENT_CHECKPOINTING_STEPS) + [0]
    for step in checkpointing_steps:
        checkpoints_directory = os.path.join(input_model_directory, f"{language}/hplt_models/bert_base_{language}")
        output_model_directory = os.path.join(output_model_directory, language+f'_{step}')
        checkpoint_path = os.path.join(
            checkpoints_directory, f"model_step_{step}.bin",
        )
        if not os.path.exists(checkpoints_directory):
            raise ValueError(f"Model directory {checkpoints_directory} does not exist")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Model file {checkpoint_path} does not exist")

        if not os.path.exists(output_model_directory):
            os.makedirs(output_model_directory)

        prototype_directory = "huggingface_prototype"
        os.system(f"cp {prototype_directory}/* {output_model_directory}")

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        torch.save(checkpoint['model'], os.path.join(output_model_directory, "pytorch_model.bin"))

        os.system(f"cp {input_model_directory}/{language}/tokenizer.json {output_model_directory}")

        tokenizer = AutoTokenizer.from_pretrained(output_model_directory)
        vocabulary_size = tokenizer.vocab_size
        config_dict = json.load(open(os.path.join(output_model_directory, "config.json")))
        config_dict["vocab_size"] = vocabulary_size
        json.dump(
            config_dict,
            open(os.path.join(output_model_directory, "config.json"), "w"),
            indent=4
        )


def main():
    args = parse_args()
    convert_to_hf(args.input_model_directory, args.output_model_directory, args.language)


if __name__ == "__main__":
    main()