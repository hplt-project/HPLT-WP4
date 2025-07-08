import json
from glob import glob

import pandas as pd

ALL_TAGS = 'AllTags'
LEMMAS = 'Lemmas'
LAS = 'LAS'

if __name__ == '__main__':
    data = {
        'lang': [],
        ALL_TAGS: [],
        LEMMAS: [],
        LAS: [],
    }
    for file_path in glob("/scratch/project_465001890/hplt-2-0-output/results/*.jsonl"):
        with open(file_path, 'r', encoding='utf8') as results_file:
            results = json.loads(results_file.readlines()[0])
            for k, v in results.items():
                data['lang'].append(k)
                data[ALL_TAGS].append(v[ALL_TAGS])
                data[LEMMAS].append(v[LEMMAS])
                data[LAS].append(v[LAS])
    df = pd.DataFrame(data)
    df.to_csv('results.csv')

