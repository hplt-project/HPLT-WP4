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


with open("old_table.txt", "r") as old:
    for line in old:
        line = [value.strip() for value in line.split("&")]

        lang = line[0].replace('\\', '')
        lang_path = f"gpt_bert_results/{lang}_hplt.jsonl"
        if os.path.exists(lang_path):
            with open(lang_path, "r") as results:
                lang_results = json.load(results)[f"{lang}_hplt"]
            # print(line)
            line.insert(5, round(lang_results["AllTags"], 1))
            # print(line)
            line.insert(10, round(lang_results["Lemmas"], 1))
            line.insert(15, round(lang_results["LAS"], 1))
            line = find_max(line, 1, 6)
            line = find_max(line,6, 11)
            line = find_max(line,11, 16)
            print(" & ".join(line).replace("\\\\", " & \\\\"))

