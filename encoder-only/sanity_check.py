import argparse
import os.path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# all samples from wikipedia
samples = {
'belC':"""
        82-я цырымонія ўручэння прэміі «Залаты глобус» — цырымонія ўзнагароджання за заслугі ў галіне кінематографа і тэлебачання за 2024 год,
         якая адбылася 5 [MASK] 2025 года. Транслявалася каналам CBS. Вядучай цырымоніі была комік Нікі Глейзер""", # https://be.wikipedia.org/wiki/%D0%97%D0%B0%D0%BB%D0%B0%D1%82%D1%8B_%D0%B3%D0%BB%D0%BE%D0%B1%D1%83%D1%81_(%D0%BF%D1%80%D1%8D%D0%BC%D1%96%D1%8F,_2025)
'itaL': """
La cerimonia di premiazione della 82ª edizione dei Golden Globe ha avuto luogo il 5 [MASK] 2025 ed è stata nuovamente trasmessa in diretta dalla rete CBS.
 È stata presentata dalla comica Nikki Glaser.
""", # https://it.wikipedia.org/wiki/Golden_Globe_2025
'cesL':    """
    82. ročník udílení Zlatých glóbů se bude konat dne 5. [MASK] 2025.
     Ceremoniál bude vysílán živě na CBS ve Spojených státech a bude ho moderovat stand-up komička Nikki Glaser.
    """,
'porL': """
A cerimónia de entrega dos prémios foi programada para ocorrer no dia 5 de [MASK] de 2025 tendo como anfitriã Nikki Glaser e transmitida em direto pela CBS.""",
    'glgL': """
    David Keith Lynch, nado en Missoula (Montana) o 20 de xaneiro de 1946 e finado nos Ánxeles o 16 de [MASK] de 2025,
     coñecido como David Lynch, foi un director de cine estadounidense. """,
"mkdC": """Така, во византискиот календар, којшто времето го мери од создавањето на светот според Библијата,
 на 1 [MASK] 5509 г. п.н.е. според продолжениот јулијански календар, истата трае во текот на 7533 и 7534 година.""",
    "nnoL": """Bulgaria og Romania blei [MASK] av Schengen-området.""",
    'islL': "Justin Trudeau segir af sér sem [MASK] Kanada.",
    'faoL': "Fyri sínar eyðsýndu listarligu gávur og tey mongu listaverkini, hon hevur evnað til hegnisliga, [MASK] Hansina Iversen Mentanarvirðisløn landsins 2022.",
    "tatL": "2025 (ике мең егерме бишенче) [MASK] — кәбисә булмаган ел, милади тәкъвим буенча чәршәмбе көнне башлана."
            " Бу безнең эраның 2025 елы, III меңьеллыкның 25 елы, XXI гасырның 25 елы, XXI гасырның 3 унъеллыгының 5 елы, 2020 елларның 6 елы.",
    'gleL': "Féile chultúrtha is ea an tOireachtas."
            " Is é an aidhm atá leis an Oireachtas ná [MASK] dúchasacha na hÉireann a chur chun cinn agus a cheiliúradh trí mheán na Gaeilge—idir amhránaíocht,"
            " ceol, rince, scéalaíocht agus drámaíocht.",
'alsL': """Thanas Papa (1931 - 3 gusht 2022) ishte një [MASK] i njohur shqiptar. 
         I lindur në fshatin Paftal të Beratit, ai është vlerësuar si i pari"
          skulptor shqiptar i diplomuar në Akademinë e Arteve të Bukura në Repin të Leningradit (sot Shën Petersburg),
           ku përfundoi studimet me medalje ari në vitin 1957.""",
    'glaL': "Bha Tutankhamun 'na [MASK] ann an 18mh rìoghrachas na h-Èipheit."
            " Bha e a’ riaghladh mu thuaiream eadar 1332 agus 1323 RC rè linn"
            " ann an eachdraidh na h-Èipheit air a bheil an Rìoghachd Ùr (no Linn na h-Impireachd Ùr uaireannan).",
    'ltzL': "Den David Keith Lynch, gebuer den 20. Januar 1946 zu Missoula, am Montana, a gestuerwen de 16. [MASK] 2025"
            " war en US-amerikanesche Kënschtler, dee virun allem fir seng Filmer bekannt war, ma och als Produzent,"
            " Dréibuchauteur, Schauspiller, Moler, Fotograf, Designer a Komponist aktiv war.",
    'mltL': "Il-Bulgarija hija [MASK] tal-Unjoni Ewropea, taż-Żona Schengen, tan-NATO u tal-Kunsill tal-Ewropa.",
    'eusL': "Zientzialari talde batek eman du [MASK]: kurlinta mokomehea desagertu da. Azken aldiz 1995ean ikusi zuten Marokon.",
}

def predict(lang):
    models_path = '/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/'
    current_model_path = os.path.join(models_path, lang)
    print(current_model_path)
    tokenizer = AutoTokenizer.from_pretrained(current_model_path)
    model = AutoModelForMaskedLM.from_pretrained(current_model_path,
                                                 trust_remote_code=True)
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    sample = samples[lang].strip()
    input_text = tokenizer(sample, return_tensors="pt")
    print(sample)
    output_p = model(**input_text)
    output_text = torch.where(input_text.input_ids == mask_id,
                              output_p.logits.argmax(-1),
                              input_text.input_ids)
    toks = output_text[0].tolist()
    print('_'.join([tokenizer.decode([tok]) for tok in toks]))

if __name__ == '__main__':
    for lang in samples.keys():
        predict(lang)
