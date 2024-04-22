import sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])

model = AutoModelForMaskedLM.from_pretrained(sys.argv[1], trust_remote_code=True)

mask_id = tokenizer.convert_tokens_to_ids("[MASK]")

text = "It's a beautiful[MASK]."

input_text = tokenizer(text, return_tensors="pt")

output_p = model(**input_text)

output_text = torch.where(input_text.input_ids == mask_id, output_p.logits.argmax(-1), input_text.input_ids)

decoded = tokenizer.decode(output_text[0].tolist())

print(text)
print(decoded)
