import os.path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

if __name__ == '__main__':
    models_path = '/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/'
    lang = 'belC'
    current_model_path = os.path.join(models_path, lang)
    tokenizer = AutoTokenizer.from_pretrained(current_model_path)
    model = AutoModelForMaskedLM.from_pretrained(current_model_path, trust_remote_code=True)
    samples = ("2024 застаўся ў [MASK]...", "І 366 дзён мы былі максімальна [MASK].", "І 366 дзён [MASK] былі максімальна сур'ёзныя")
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    for sample in samples:
        input_text = tokenizer(sample, return_tensors="pt")
        output_p = model(**input_text)
        output_text = torch.where(input_text.input_ids == mask_id, output_p.logits.argmax(-1), input_text.input_ids)

        print(tokenizer.decode(output_text[0].tolist()))