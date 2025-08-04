import torch
from safetensors.torch import save_file, load_file
import os
import json
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
from huggingface_hub import get_collection, HfApi, hf_hub_download


def convert_pytorch_to_safetensors(pytorch_path: str, safetensors_path: str):
    print(f"Loading PyTorch checkpoint from: {pytorch_path}")

    # Load the PyTorch checkpoint
    try:
        checkpoint = torch.load(pytorch_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch checkpoint: {e}")
    
    # Extract tensor weights
    state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    if not state_dict:
        raise ValueError("No tensors found in the checkpoint")
    
    # Detect shared tensors
    tensor_to_keys = {}
    shared_tensors = []
    shared_groups = {}
    for key, tensor in state_dict.items():
        data_ptr = tensor.data_ptr()
        if data_ptr in tensor_to_keys:
            shared_tensors.append((key, tensor_to_keys[data_ptr]))
            if data_ptr not in shared_groups:
                shared_groups[data_ptr] = [tensor_to_keys[data_ptr]]
            shared_groups[data_ptr].append(key)
        else:
            tensor_to_keys[data_ptr] = key
    
    if shared_tensors:
        print(f"\nDetected shared tensors:")
        for data_ptr, keys in shared_groups.items():
            print(f"  - These tensors share memory: {keys}")
        print("These will have to be cloned for conversion to safetensors.")
    
    # Convert tensors to contiguous format (required by safetensors)
    processed_state_dict = {}
    for key, tensor in state_dict.items():
        # Clone the tensor to ensure it has its own memory
        processed_state_dict[key] = tensor.clone().contiguous()

    # Save to safetensors format
    print(f"\nSaving to safetensors format: {safetensors_path}")
    save_file(processed_state_dict, safetensors_path)
    
    # Print summary
    print(f"\nConversion complete!")
    print(f"Total number of converted parameters: {sum(p.numel() for p in processed_state_dict.values())}")
    print(f"Input file size: {os.path.getsize(pytorch_path) / 1024 / 1024:.2f} MB")
    print(f"Output file size: {os.path.getsize(safetensors_path) / 1024 / 1024:.2f} MB")


def verify_conversion(pytorch_path: str, safetensors_path: str):
    print("\nVerifying conversion...")
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    pt_state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}

    # Load safetensors file
    st_state_dict = load_file(safetensors_path)
    
    # Compare keys
    pt_keys = set(k for k, v in pt_state_dict.items() if isinstance(v, torch.Tensor))
    st_keys = set(st_state_dict.keys())

    if pt_keys != st_keys:
        print(f"Key mismatch! PyTorch keys: {len(pt_keys)}, Safetensors keys: {len(st_keys)}")
        missing_in_st = pt_keys - st_keys
        extra_in_st = st_keys - pt_keys
        if missing_in_st:
            print(f"Missing in safetensors: {missing_in_st}")
        if extra_in_st:
            print(f"Extra in safetensors: {extra_in_st}")
        return False
    
    # Compare tensor values
    all_match = True
    for key in pt_keys:
        pt_tensor = pt_state_dict[key].cpu()
        st_tensor = st_state_dict[key]
        
        if not torch.allclose(pt_tensor, st_tensor, rtol=1e-5, atol=1e-5):
            print(f"Tensor mismatch for key: {key}")
            all_match = False
    
    if all_match:
        print("🔥 All tensors match! Conversion successful.")


@torch.no_grad()
def show_certainty(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors='pt')["input_ids"]
    N = tokens.size(1)
    tokens = tokens.repeat(N - 2, 1)
    mask = torch.eye(N).bool()[1:-1, :]
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    bert_input = tokens.masked_fill(mask, value=mask_id)

    words = [tokenizer.decode([tokens[0, i + 1].item()]) for i in range(N-2)]
    max_word_len = max(max(len(w) for w in words), 5) + 3
    print('ORIGINAL:', end='')
    for word in words:
        print((max_word_len - len(word)) * ' ' + word, end='')
    print()

    logits = model(input_ids=bert_input)["logits"]
    log_p = F.log_softmax(logits, dim=-1)
    log_p = log_p.gather(index=tokens.unsqueeze(-1), dim=-1).squeeze(-1)
    log_p = log_p.masked_fill(~mask, 0.0).sum(-1)

    print("PROB:    ", end='')
    for i in range(N-2):
        word_id = tokens[0, i+1].item()
        s = f"{log_p[i].exp().item() * 100:02.2f}"
        print((max_word_len - len(s)) * ' ' + s, end='')

    print()
    print("ARGMAX:  ", end='')
    for i in range(N-2):
        max_word = tokenizer.decode(logits[i, i+1, :].argmax())
        print((max_word_len - len(max_word)) * ' ' + max_word, end='')

    print(f"\nSENTENCE PPL:   {log_p.sum()}\n")


parser = argparse.ArgumentParser()
parser.add_argument('--collection_slug', default='HPLT/hplt-20-bert-models-67ba52ae96b1fb8aae673493')
args = parser.parse_args()

api = HfApi()
collection = get_collection(args.collection_slug)
safetensors_path = 'model.safetensors'
fn = 'pytorch_model.bin'
for item in collection.items:
    print(item.item_id)
    refs = api.list_repo_refs(item.item_id)
    for branch in refs.branches:
        hf_hub_download(repo_id=item.item_id, filename=fn, revision=branch.name, local_dir='.')
        convert_pytorch_to_safetensors(pytorch_path=fn, safetensors_path=safetensors_path)
        verify_conversion(pytorch_path=fn, safetensors_path=safetensors_path)
        api.upload_file(
            path_or_fileobj=safetensors_path,
            path_in_repo=safetensors_path,
            repo_id=item.item_id,
            repo_type="model",
            commit_message=f"Upload {safetensors_path}",
            revision=branch.name,
        )