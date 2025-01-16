import argparse
import os.path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

samples = {
'belC':"""
        82-я цырымонія ўручэння прэміі «Залаты глобус» — цырымонія ўзнагароджання за заслугі ў галіне кінематографа і тэлебачання за 2024 [MASK],
         якая адбылася 5 студзеня 2025 года. Транслявалася каналам CBS. Вядучай цырымоніі была комік Нікі Глейзер""", # https://be.wikipedia.org/wiki/%D0%97%D0%B0%D0%BB%D0%B0%D1%82%D1%8B_%D0%B3%D0%BB%D0%BE%D0%B1%D1%83%D1%81_(%D0%BF%D1%80%D1%8D%D0%BC%D1%96%D1%8F,_2025)
'itaL': """
La cerimonia di premiazione della 82ª edizione dei Golden Globe ha avuto luogo il 5 [MASK] 2025 ed è stata nuovamente trasmessa in diretta dalla rete CBS.
 È stata presentata dalla comica Nikki Glaser.
""", # https://it.wikipedia.org/wiki/Golden_Globe_2025

}

def predict(lang):
    models_path = '/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/'
    current_model_path = os.path.join(models_path, lang)
    tokenizer = AutoTokenizer.from_pretrained(current_model_path)
    model = AutoModelForMaskedLM.from_pretrained(current_model_path,
                                                 trust_remote_code=True)
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    input_text = tokenizer(samples[lang].strip(), return_tensors="pt")
    print(input_text.input_ids)
    output_p = model(**input_text)
    output_text = torch.where(input_text.input_ids == mask_id,
                              output_p.logits.argmax(-1),
                              input_text.input_ids)
    toks = output_text[0].tolist()
    print('_'.join([tokenizer.decode([tok]) for tok in toks]))

if __name__ == '__main__':
    for lang in ('belC', 'itaL'):
        predict(lang)
