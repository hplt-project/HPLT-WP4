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

# HPLT Bert for English

<img src="https://hplt-project.org/_next/static/media/logo-hplt.d5e16ca5.svg" width=12.5%>

This is one of the encoder-only monolingual language models trained as a second release by the [HPLT project](https://hplt-project.org/).
It is a so called masked language model. In particular, we used the modification of the classic BERT model named [LTG-BERT](https://aclanthology.org/2023.findings-eacl.146/).

A monolingual LTG-BERT model is trained for some languages in the [HPLT 2.0 data release](https://hplt-project.org/datasets/v2.0).

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
})
```

```bibtex
@inproceedings{de-gibert-etal-2024-new-massive,
    title = "A New Massive Multilingual Dataset for High-Performance Language Technologies",
    author = {de Gibert, Ona  and
      Nail, Graeme  and
      Arefyev, Nikolay  and
      Ba{\~n}{\'o}n, Marta  and
      van der Linde, Jelmer  and
      Ji, Shaoxiong  and
      Zaragoza-Bernabeu, Jaume  and
      Aulamo, Mikko  and
      Ram{\'\i}rez-S{\'a}nchez, Gema  and
      Kutuzov, Andrey  and
      Pyysalo, Sampo  and
      Oepen, Stephan  and
      Tiedemann, J{\"o}rg},
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.100",
    pages = "1116--1128",
    abstract = "We present the HPLT (High Performance Language Technologies) language resources, a new massive multilingual dataset including both monolingual and bilingual corpora extracted from CommonCrawl and previously unused web crawls from the Internet Archive. We describe our methods for data acquisition, management and processing of large corpora, which rely on open-source software tools and high-performance computing. Our monolingual collection focuses on low- to medium-resourced languages and covers 75 languages and a total of {\mbox{$\approx$}} 5.6 trillion word tokens de-duplicated on the document level. Our English-centric parallel corpus is derived from its monolingual counterpart and covers 18 language pairs and more than 96 million aligned sentence pairs with roughly 1.4 billion English tokens. The HPLT language resources are one of the largest open text corpora ever released, providing a great resource for language modeling and machine translation training. We publicly release the corpora, the software, and the tools used in this work.",
}
```
