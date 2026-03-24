from argparse import Namespace

from transformers import AutoTokenizer
import torch

from model import Model


def load_model(model_path, device="cpu"):
    args = Namespace()
    args.model_path = model_path
    model = torch.load("model.bin", map_location=torch.device(device))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    args.dropout = 0.3
    model["dataset"] = Namespace(**model["dataset"])
    predict_model = Model(args, model["dataset"])
    predict_model.load_state_dict(model["model"])
    return tokenizer, model, predict_model

