---
language:
- en
- eng
inference: false
tags:
- T5
- t5
- HPLT
- encoder-decoder
- text2text-generation
license: apache-2.0
datasets:
- HPLT/HPLT3.0
---

# HPLT v3.0 T5 for English

<img src="https://hplt-project.org/_next/static/media/logo-hplt.d5e16ca5.svg" width=12.5%>

This is one of the encoder-decoder monolingual language models trained as a third release by the [HPLT project](https://hplt-project.org/).
It is a text-to-text transformer trained with a denoising objective. Our
models follow the setup of [NorT5](https://aclanthology.org/2023.nodalida-1.61/).

We present monolingual NorT5 models for 57 languages out of 198 total in the [HPLT v3.0 dataset](https://hplt-project.org/datasets/v3.0).

All the HPLT encoder-decoder models use the same hyper-parameters, roughly following the T5-base setup:
- hidden size: 768
- attention heads: 12
- layers: 12
- vocabulary size: 32768

Every model uses its own tokenizer trained on language-specific HPLT data. 

[The training code](https://github.com/hplt-project/HPLT-WP4).

## Example usage

This model currently needs a custom wrapper from `modeling_nort5.py`, you should therefore load the model with `trust_remote_code=True`.

```
pip install transformers==4.46.1
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = 'HPLT/hplt_t5_base_3_0_nob_Latn'
model = AutoModelForSeq2SeqLM.from_pretrained(
  model_path, trust_remote_code=True, use_safetensors=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# MASKED LANGUAGE MODELING
sentence = "Ansiktsuttrykket [MASK_1] har utviklet seg til et utbredt kulturelt fenomen."
encoding = tokenizer(sentence, return_tensors="pt")
mask_1 = tokenizer.convert_tokens_to_ids("[MASK_1]")
mask_2 = tokenizer.convert_tokens_to_ids("[MASK_2]")
output_tensor = model.generate(
    encoding.input_ids,
    decoder_start_token_id=mask_1,
    eos_token_id=mask_2,
  )
print(tokenizer.decode(output_tensor.squeeze(), skip_special_tokens=False))
# should output: '[MASK_1]«The Great Gatsby»[MASK_2]'
```

## Intermediate checkpoints

We are releasing 10 intermediate checkpoints for each model at intervals of every 3125 training steps in separate branches. The naming convention is `stepXXX`: for example, `step18750`.

You can load a specific model revision with `transformers` using the argument `revision`:
```python
model = AutoModelForSeq2SeqLM.from_pretrained("HPLT/hplt_t5_base_3_0_eng_Latn", revision="step21875", trust_remote_code=True)
```

You can access all the revisions for the models with the following code:
```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("HPLT/hplt_t5_base_3_0_eng_Latn")
print([b.name for b in out.branches])
```

## Cite us

```bibtex
@inproceedings{samuel-etal-2023-norbench,
    title = "{N}or{B}ench {--} A Benchmark for {N}orwegian Language Models",
    author = "Samuel, David  and
      Kutuzov, Andrey  and
      Touileb, Samia  and
      Velldal, Erik  and
      {\O}vrelid, Lilja  and
      R{\o}nningstad, Egil  and
      Sigdel, Elina  and
      Palatkina, Anna",
    editor = {Alum{\"a}e, Tanel  and
      Fishel, Mark},
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.61/",
    pages = "618--633",
    abstract = "We present NorBench: a streamlined suite of NLP tasks and probes for evaluating Norwegian language models (LMs) on standardized data splits and evaluation metrics. We also introduce a range of new Norwegian language models (both encoder and encoder-decoder based). Finally, we compare and analyze their performance, along with other existing LMs, across the different benchmark tests of NorBench."
}
```

```bibtex
@inproceedings{burchell-etal-2025-expanded,
    title = "An Expanded Massive Multilingual Dataset for High-Performance Language Technologies ({HPLT})",
    author = {Burchell, Laurie  and
      de Gibert, Ona  and
      Arefyev, Nikolay  and
      Aulamo, Mikko  and
      Ba{\~n}{\'o}n, Marta  and
      Chen, Pinzhen  and
      Fedorova, Mariia  and
      Guillou, Liane  and
      Haddow, Barry  and
      Haji{\v{c}}, Jan  and
      Helcl, Jind{\v{r}}ich  and
      Henriksson, Erik  and
      Klimaszewski, Mateusz  and
      Komulainen, Ville  and
      Kutuzov, Andrey  and
      Kyt{\"o}niemi, Joona  and
      Laippala, Veronika  and
      M{\ae}hlum, Petter  and
      Malik, Bhavitvya  and
      Mehryary, Farrokh  and
      Mikhailov, Vladislav  and
      Moghe, Nikita  and
      Myntti, Amanda  and
      O{'}Brien, Dayy{\'a}n  and
      Oepen, Stephan  and
      Pal, Proyag  and
      Piha, Jousia  and
      Pyysalo, Sampo  and
      Ram{\'i}rez-S{\'a}nchez, Gema  and
      Samuel, David  and
      Stepachev, Pavel  and
      Tiedemann, J{\"o}rg  and
      Vari{\v{s}}, Du{\v{s}}an  and
      Vojt{\v{e}}chov{\'a}, Tereza  and
      Zaragoza-Bernabeu, Jaume},
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.854/",
    doi = "10.18653/v1/2025.acl-long.854",
    pages = "17452--17485",
    ISBN = "979-8-89176-251-0",
    abstract = "Training state-of-the-art large language models requires vast amounts of clean and diverse textual data. However, building suitable multilingual datasets remains a challenge. In this work, we present HPLT v2, a collection of high-quality multilingual monolingual and parallel corpora, extending prior work of the HPLT project. The monolingual portion of the data contains 8T tokens covering 193 languages, while the parallel data contains 380M sentence pairs covering 51 languages. We document the entire data pipeline and release the code to reproduce it. We provide extensive analysis of the quality and characteristics of our data. Finally, we evaluate the performance of language models and machine translation systems trained on HPLT v2, demonstrating its value."
}
```
