import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

test = "David Keith Lynch (Januari 20, 1946- Januari 15, 2025) alizaliwa huko Missoula, Montana, Marekani." # https://sw.wikipedia.org/wiki/David_Lynch
model_path = '/scratch/project_465001925/mariiaf/swh-Latn-t5'
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# MASKED LANGUAGE MODELING
sentence = "David Keith Lynch (Januari 20, 1946-[MASK] 15, 2025) alizaliwa huko Missoula, Montana, Marekani."
encoding = tokenizer(sentence)

input_tensor = torch.tensor([encoding.input_ids])
output_tensor = model.generate(input_tensor)
print(tokenizer.decode(output_tensor.squeeze(), skip_special_tokens=False))
