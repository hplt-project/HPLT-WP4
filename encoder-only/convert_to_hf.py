import os
import torch
import argparse
import json
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_directory', type=str, default='/scratch/project_465000498/hplt_models')
    parser.add_argument('--output_model_directory', type=str, default='/scratch/project_465000498/hplt_hf_models')
    parser.add_argument('--language', type=str, default='en')
    args = parser.parse_args()
    return args


def convert_to_hf(input_model_directory, output_model_directory, language):
    input_model_directory = os.path.join(input_model_directory, f"bert_base_{language}")
    output_model_directory = os.path.join(output_model_directory, language)

    if not os.path.exists(input_model_directory):
        raise ValueError(f"Model directory {input_model_directory} does not exist")
    if not os.path.exists(os.path.join(input_model_directory, "model_step_31250.bin")):
        raise ValueError(f"Model file {os.path.join(input_model_directory, 'model_step_31250.bin')} does not exist")

    if not os.path.exists(output_model_directory):
        os.makedirs(output_model_directory)

    prototype_directory = "huggingface_prototype"
    os.system(f"cp {prototype_directory}/* {output_model_directory}")

    checkpoint = torch.load(os.path.join(input_model_directory, "model_step_31250.bin"))
    torch.save(checkpoint['model'], os.path.join(output_model_directory, "pytorch_model.bin"))

    os.system(f"cp /scratch/project_465000498/processed_data/{language}/tokenizer.json {output_model_directory}")

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