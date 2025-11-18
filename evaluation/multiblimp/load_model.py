import tempfile
from typing import *

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from minicons import scorer


# todo: take script from the actual MultiBLiMP file, not in this order
SCRIPTS = [
    "arab",
    "beng",
    "cyrl",
    "ethi",
    "tibt",
    "thaa",
    "grek",
    "hebr",
    "deva",
    "armn",
    "cans",
    "latn",
]
SIZES = [
    "5mb",
    "10mb",
    "100mb",
    "1000mb",
]


def load_hf_model(model_name: str, no_cache=False, is_encoder=False, **kwargs):
    model = None
    tokenizer = None

    if no_cache:
        with tempfile.TemporaryDirectory() as tmpdirname:
            if not is_encoder:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=tmpdirname, **kwargs
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, cache_dir=tmpdirname, trust_remote_code=True, **kwargs,
                )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=tmpdirname, **kwargs
            )
    else:
        if not is_encoder:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True, **kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if 'hplt' in model_name:
        model.config.decoder_start_token_id = 4
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    if model is None:
        print("Model is None")
        return None
    if not is_encoder:
        ilm_model = scorer.IncrementalLMScorer(
            model,
            "cuda",
            tokenizer=tokenizer,
        )
    else:
        ilm_model = scorer.Seq2SeqScorer(model, tokenizer=tokenizer, device="cuda")

    return ilm_model
