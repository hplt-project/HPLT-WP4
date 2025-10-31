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
import torch
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
model = AutoModelForMaskedLM.from_pretrained("HPLT/hplt_t5_base_3_0_eng_Latn", revision="step21875", trust_remote_code=True)
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
