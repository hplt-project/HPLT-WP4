import json
from operator import itemgetter
import os.path
import iso639
import pandas as pd

MAPPING = {
    "zho_Hans": "cmn_Hans",
    "est_Latn": "ekk_Latn",
}


def find_max(line, start, end):
    sublist = [
        round(float(result.removeprefix('\\textbf{').rstrip("}")), 1) if isinstance(result, str) else result
        for result in line[start:end]
    ]
    max_pos, max_result = max(enumerate(sublist), key=itemgetter(1))
    sublist[max_pos] = "\\textbf{" + str(max_result) + "}"
    sublist = [str(result) for result in sublist]
    line[start:end] = sublist
    return line


def insert_model_results(line, lang_path, start, step, model, lang):
    with open(lang_path, "r") as results:
        lang_results = json.load(results)[f"{lang}_{model}"]
    # print(line)
    line.insert(start, round(lang_results["AllTags"], 1))
    # print(line)
    line.insert(start + step, round(lang_results["Lemmas"], 1))
    line.insert(start + step*2, round(lang_results["LAS"], 1))
    return line


def insert(lang_path, lang_path_2, line, lang, counter):
    line = insert_model_results(line, lang_path, 5, 5, "hplt", lang)
    line = insert_model_results(line, lang_path_2, 3, 6, "mmbert", lang)
    line = find_max(line, 1, 7)
    line = find_max(line, 7, 13)
    line = find_max(line, 13, 19)
    line[0] = lang.replace("_", "\_")
    print(" & ".join(line) + "\\\\")
    return counter + 1


with open("old_table.txt", "r") as old:
    counter = 0
    ner = pd.read_csv("../ner/ner_results.tsv", sep="\t", header=None)
    for line in old:
        line = [value.strip() for value in line.split("&")]
        line[-1] = line[-1].removesuffix(' \\\\')
        lang = line[0].replace('\\', '')
        lang_path = f"gpt_bert_results/{lang}_hplt.jsonl"
        lang_path_2 = f"ud_results_mmbert/{lang}_mmbert.jsonl"

        ner_mmbert = ner[ner[0] == "jhu-clsp/mmBERT-base"]
        lang_object = iso639.Language.from_part3(lang[:3])
        pt1 = lang_object.part1
        if os.path.exists(lang_path) and os.path.exists(lang_path_2):
            lang_ner = ner[ner[0] == f"HPLT/hplt_gpt_bert_base_3_0_{lang}"][2]
            if lang_ner.shape[0] > 0:
                lang_ner = str(round(lang_ner.max() * 100, 1))
            else:
                lang_ner = " "
            line.append(lang_ner)

            mmbert_ner = ner_mmbert[ner_mmbert[1] == f"wikiann/{pt1}"][2]
            if mmbert_ner.shape[0] > 0:
                mmbert_ner = str(round(mmbert_ner.max() * 100, 1))
            else:
                mmbert_ner = " "
            line.insert(15, mmbert_ner)

            counter = insert(lang_path, lang_path_2, line, lang, counter)
        else:
            new_lang = MAPPING.get(lang)
            if new_lang is not None:
                lang_path = f"gpt_bert_results/{new_lang}_hplt.jsonl"
                lang_path_2 = f"ud_results_mmbert/{new_lang}_mmbert.jsonl"
                lang_ner = ner[ner[0] == f"HPLT/hplt_gpt_bert_base_3_0_{new_lang}"][2]
                if lang_ner.shape[0] > 0:
                    lang_ner = str(round(lang_ner.max() * 100, 1))
                else:
                    lang_ner = " "
                line.append(lang_ner)

                mmbert_ner = ner_mmbert[ner_mmbert[1] == f"wikiann/{pt1}"][2]
                if mmbert_ner.shape[0] > 0:
                    mmbert_ner = str(round(mmbert_ner.max() * 100, 1))
                else:
                    mmbert_ner = " "
                line.insert(15, mmbert_ner)

                counter = insert(lang_path, lang_path_2, line, new_lang, counter)
    print(counter)


