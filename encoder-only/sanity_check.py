import os.path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

if __name__ == '__main__':
    models_path = '/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/'
    lang = 'belC'
    current_model_path = os.path.join(models_path, lang)
    tokenizer = AutoTokenizer.from_pretrained(current_model_path)
    model = AutoModelForMaskedLM.from_pretrained(current_model_path, trust_remote_code=True)
    samples = ("""
    82-я цырымонія ўручэння прэміі «Залаты глобус» [MASK] цырымонія ўзнагароджання [MASK] заслугі [MASK] галіне кінематографа [MASK] тэлебачання за 2024 [MASK],
     якая адбылася 5 студзеня 2025 года. Транслявалася каналам CBS. Вядучай цырымоніі была комік Нікі Глейзер.
    """) # https://be.wikipedia.org/wiki/%D0%97%D0%B0%D0%BB%D0%B0%D1%82%D1%8B_%D0%B3%D0%BB%D0%BE%D0%B1%D1%83%D1%81_(%D0%BF%D1%80%D1%8D%D0%BC%D1%96%D1%8F,_2025)
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    for sample in samples:
        input_text = tokenizer(sample, return_tensors="pt")
        print(input_text.input_ids)
        output_p = model(**input_text)
        output_text = torch.where(input_text.input_ids == mask_id, output_p.logits.argmax(-1), input_text.input_ids)

        print(tokenizer.decode(output_text[0].tolist()))