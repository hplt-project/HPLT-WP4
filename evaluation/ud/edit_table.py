import json
from operator import itemgetter
import os.path


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


def insert_model_results(lang_path, start, step, model):
    with open(lang_path, "r") as results:
        lang_results = json.load(results)[f"{lang}_{model}"]
    # print(line)
    line.insert(start, round(lang_results["AllTags"], 1))
    # print(line)
    line.insert(start + step, round(lang_results["Lemmas"], 1))
    line.insert(start + step*2, round(lang_results["LAS"], 1))
    return line


with open("old_table.txt", "r") as old:
    for line in old:
        line = [value.strip() for value in line.split("&")]

        lang = line[0].replace('\\', '')
        lang_path = f"gpt_bert_results/{lang}_hplt.jsonl"
        lang_path_2 = f"ud_results_mmbert/{lang}_mmbert.jsonl"
        if os.path.exists(lang_path) and os.path.exists(lang_path_2):
            line = insert_model_results(lang_path, 5, 5, "hplt")
            line = insert_model_results(lang_path_2, 6, 6, "mmbert")
            line = find_max(line, 1, 7)
            line = find_max(line,7, 13)
            line = find_max(line,13, 19)
            print(" & ".join(line).replace("\\\\", " & & \\\\"))

