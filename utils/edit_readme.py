#! /bin/env/python3

import json
import os

with open("ISO-639-1-language.json") as f:
    a = json.load(f)

languages = {}

for el in a:
    code = el["code"]
    name = el["name"]
    languages[code] = name

# print(languages)

for el in os.listdir("."):
    if os.path.isdir(el) and el.startswith("hplt") and not "hbs" in el:
        langcode = el.split("_")[-1]
        langname = languages[langcode]
        print(el, langcode, langname)
        with open(os.path.join(el, "README.md"), 'r') as file:
            contents = file.read()
        # Replace the word
        new_contents = contents.replace("English", langname)
        with open(os.path.join(el, "README.md"), 'w') as file:
            file.write(new_contents)
