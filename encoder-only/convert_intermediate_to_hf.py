import os
import torch
import argparse
import json
from transformers import AutoTokenizer
import logging
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model_directory",
        type=str,
        default="/scratch/project_465000498/hplt_models",
    )
    parser.add_argument(
        "--output_model_directory",
        type=str,
        default="/scratch/project_465000498/hplt_hf_models/intermediate",
    )
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()
    return args


def convert_hf(input_model_directory, output_model_directory, language, template_dir):
    input_model_directory = os.path.join(input_model_directory, f"bert_base_{language}")
    output_model_directory = os.path.join(
        output_model_directory, f"hplt_bert_base_{language}-intermediate"
    )

    if not os.path.exists(input_model_directory):
        raise ValueError(f"Model directory {input_model_directory} does not exist")

    if not os.path.exists(output_model_directory):
        os.makedirs(output_model_directory)

    checkpoints = [
        el.name
        for el in os.scandir(input_model_directory)
        if el.is_file() and el.name.endswith(".bin")
    ]

    for checkpoint in checkpoints:
        logger.info(f"Processing checkpoint {checkpoint}...")
        step_id = "step" + os.path.splitext(checkpoint)[0].split("_")[2]
        step_dir = os.path.join(output_model_directory, step_id)
        _ = shutil.copytree(template_dir, step_dir)

        cur_checkpoint = torch.load(
            os.path.join(input_model_directory, checkpoint),
            map_location=torch.device("cpu"),
        )
        torch.save(cur_checkpoint["model"], os.path.join(step_dir, "pytorch_model.bin"))

        _ = shutil.copy2(
            f"/scratch/project_465000498/processed_data/{language}/tokenizer.json",
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


def main():
    args = parse_args()
    prototype_directory = "huggingface_prototype"
    convert_hf(
        args.input_model_directory,
        args.output_model_directory,
        args.language,
        prototype_directory,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    main()
