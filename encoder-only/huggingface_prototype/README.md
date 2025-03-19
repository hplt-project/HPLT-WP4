---
language:
- en
inference: false
tags:
- BERT
- HPLT
- encoder
license: apache-2.0
datasets:
- HPLT/HPLT2.0_cleaned
---

# HPLT v2.0 for English

<img src="https://hplt-project.org/_next/static/media/logo-hplt.d5e16ca5.svg" width=12.5%>

This is one of the encoder-only monolingual language models trained as a second release by the [HPLT project](https://hplt-project.org/).
It is a so called masked language model. In particular, we used the modification of the classic BERT model named [LTG-BERT](https://aclanthology.org/2023.findings-eacl.146/).

We present monolingual LTG-BERT models for more than 50 languages out of 191 total in the [HPLT v2.0 dataset](https://hplt-project.org/datasets/v2.0).

All the HPLT encoder-only models use the same hyper-parameters, roughly following the BERT-base setup:
- hidden size: 768
- attention heads: 12
- layers: 12
- vocabulary size: 32768

Every model uses its own tokenizer trained on language-specific HPLT data. 

[The training code](https://github.com/hplt-project/HPLT-WP4).

[The training statistics of all runs](https://api.wandb.ai/links/ltg/kduj7mjn)

## Example usage

This model currently needs a custom wrapper from `modeling_ltgbert.py`, you should therefore load the model with `trust_remote_code=True`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("HPLT/hplt_bert_base_eng-Latn")
model = AutoModelForMaskedLM.from_pretrained("HPLT/hplt_bert_base_eng-Latn", trust_remote_code=True)

mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
input_text = tokenizer("It's a beautiful[MASK].", return_tensors="pt")
output_p = model(**input_text)
output_text = torch.where(input_text.input_ids == mask_id, output_p.logits.argmax(-1), input_text.input_ids)

# should output: '[CLS] It's a beautiful place.[SEP]'
print(tokenizer.decode(output_text[0].tolist()))
```

The following classes are currently implemented: `AutoModel`, `AutoModelMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering` and `AutoModeltForMultipleChoice`.

## Intermediate checkpoints

We are releasing 10 intermediate checkpoints for each model at intervals of every 3125 training steps in separate branches. The naming convention is `stepXXX`: for example, `step18750`.

You can load a specific model revision with `transformers` using the argument `revision`:
```python
model = AutoModelForMaskedLM.from_pretrained("HPLT/hplt_bert_base_eng-Latn", revision="step21875", trust_remote_code=True)
```

You can access all the revisions for the models with the following code:
```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("HPLT/hplt_bert_base_eng-Latn")
print([b.name for b in out.branches])
```

## Cite us

```bibtex
@inproceedings{samuel-etal-2023-trained,
    title = "Trained on 100 million words and still in shape: {BERT} meets {B}ritish {N}ational {C}orpus",
    author = "Samuel, David  and
      Kutuzov, Andrey  and
      {\O}vrelid, Lilja  and
      Velldal, Erik",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.146",
    doi = "10.18653/v1/2023.findings-eacl.146",
    pages = "1954--1974"
}
```

```bibtex
@misc{burchell2025expandedmassivemultilingualdataset,
      title={An Expanded Massive Multilingual Dataset for High-Performance Language Technologies},
      author={Laurie Burchell and Ona de Gibert and Nikolay Arefyev and Mikko Aulamo and Marta Bañón and Pinzhen Chen and Mariia Fedorova and Liane Guillou and Barry Haddow and Jan Hajič and Jindřich Helcl and Erik Henriksson and Mateusz Klimaszewski and Ville Komulainen and Andrey Kutuzov and Joona Kytöniemi and Veronika Laippala and Petter Mæhlum and Bhavitvya Malik and Farrokh Mehryary and Vladislav Mikhailov and Nikita Moghe and Amanda Myntti and Dayyán O'Brien and Stephan Oepen and Proyag Pal and Jousia Piha and Sampo Pyysalo and Gema Ramírez-Sánchez and David Samuel and Pavel Stepachev and Jörg Tiedemann and Dušan Variš and Tereza Vojtěchová and Jaume Zaragoza-Bernabeu},
      year={2025},
      eprint={2503.10267},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.10267},
}
```
