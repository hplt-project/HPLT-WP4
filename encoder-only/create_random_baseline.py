import argparse
import os

from tokenizers import Tokenizer
import torch

from config import BertConfig
from model import Bert
from utils import seed_everything

SEED = 42

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="ellG", type=str)
    parser.add_argument(
        "--input_dir",
                        default="/scratch/project_465001386/hplt-2-0-output",
                        type=str,
                        )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    rank = int(os.getenv("SLURM_PROCID")) # most likely 0 on LUMI?
    seed_everything(SEED + rank)
    config = BertConfig('configs/base.json')
    lang_dir = os.path.join(args.input_dir, args.language)
    tokenizer_path = os.path.join(lang_dir, 'tokenizer.json')
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    config.vocab_size = tokenizer.get_vocab_size()
    model = Bert(config)
    checkpoint_path = f"{lang_dir}/hplt_models/bert_base_{args.language}/model_step_0.bin"
    model_to_save = model.module if hasattr(
        model, 'module',
    ) else model  # Only save the model itself
    torch.save(
        {
            "model": model_to_save.state_dict(),
            "global_step": 0,
            "epoch": 0,
            "args": args,
        },
        checkpoint_path,
    )