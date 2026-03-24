---
language:
- en
- eng
inference: false
tags:
- BERT
- HPLT
- token-classification
license: apache-2.0
datasets:
- HPLT/HPLT3.0
---

# HPLT v3.0 GPT-BERT UD parser for English

<img src="https://hplt-project.org/_next/static/media/logo-hplt.d5e16ca5.svg" width=12.5%>

This is an [HPLT](https://hplt-project.org/) [GPT-BERT](https://aclanthology.org/2024.conll-babylm.24/) model fine-tuned on [Universal Dependencies](https://universaldependencies.org/).

It performs POS-tagging, lemmatization and syntactic parsing.

[The training code](https://github.com/hplt-project/HPLT-WP4/tree/main/evaluation/ud).

```
pip install transformers==4.57.6 ufal.chu_liu_edmonds==1.0.3
```
## Example usage

```shell
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/HPLT/hplt_gpt_bert_base_3_0_eng_Latn-UD
cd hplt_gpt_bert_base_3_0_eng_Latn-UD/
```

```python
import torch

from lemma_rule import apply_lemma_rule
from preprocessor import Preprocessor
from utils import load_model


sentences = [
    "One evening I was walking along a path, the city was on one side and the fjord below.",
    "I stopped and looked out over the fjord – the sun was setting, and the clouds turning blood red.",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model, predict_model = load_model(model_path="HPLT/hplt_gpt_bert_base_3_0_eng_Latn", device=device)
predict_model.eval()
preprocessor = Preprocessor(tokenizer=tokenizer)
batch = preprocessor.preprocess(sentences)
with torch.no_grad():
    lemma_p, upos_p, xpos_p, feats_p, _, __, dep_p, head_p = predict_model(
                                batch["subwords"],
                                batch["alignment"],
                                batch["subword_lengths"],
                                batch["word_lengths"],
                            )
for i in range(len(sentences)):
    for j, form in enumerate(batch["words"][i]):

        lemma_rule = {
            rule_type: model["dataset"].lemma_vocab[rule_type].get(p[i, j, :].argmax().item(),
                       model["dataset"].lemma_vocab[rule_type][-1])
            for rule_type, p in lemma_p.items()
        }
        print(form)
        print(f"lemma {apply_lemma_rule(form, lemma_rule)}")
        print(f"upos {model['dataset'].upos_vocab[upos_p[i, j, :].argmax().item()]}")
        print(f"xpos {model['dataset'].xpos_vocab[xpos_p[i, j, :].argmax().item()]}")
        print(f"feats {model['dataset'].feats_vocab[feats_p[i, j, :].argmax().item()]}")
        print(f"head {head_p[i, j].item()}")
        print(f"deprel {model['dataset'].arc_dep_vocab[dep_p[i, j, :].argmax().item()]}")
        print("_____________________________________________________________________")
```

Should output:

```
One
lemma one
upos NUM
xpos CD
feats NumForm=Word|NumType=Card
head 2
deprel nummod
_____________________________________________________________________
evening
lemma evening
upos NOUN
xpos NN
feats Number=Sing
head 5
deprel obl:tmod
_____________________________________________________________________
```

... and so on.

## Cite us

```bibtex
@inproceedings{charpentier-samuel-2024-bert,
    title = "{GPT} or {BERT}: why not both?",
    author = "Charpentier, Lucas Georges Gabriel  and
      Samuel, David",
    booktitle = "The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conll-babylm.24/",
    pages = "262--283"
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

```bibtex
@misc{oepen2025hplt30largescalemultilingual,
      title={{HPLT 3.0}: {V}ery Large-Scale Multilingual Resources for {LLM} and {MT}. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models}, 
      author={Stephan Oepen and Nikolay Arefev and Mikko Aulamo and Marta Bañón and Maja Buljan and Laurie Burchell and Lucas Charpentier and Pinzhen Chen and Mariia Fedorova and Ona de Gibert and Barry Haddow and Jan Hajič and Jindřich Helcl and Andrey Kutuzov and Veronika Laippala and Zihao Li and Risto Luukkonen and Bhavitvya Malik and Vladislav Mikhailov and Amanda Myntti and Dayyán O'Brien and Lucie Poláková and Sampo Pyysalo and Gema Ramírez Sánchez and Janine Siewert and Pavel Stepachev and Jörg Tiedemann and Teemu Vahtola and Dušan Variš and Fedor Vitiugin and Tea Vojtěchová and Jaume Zaragoza},
      year={2025},
      eprint={2511.01066},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01066}, 
}
```

## Cite UD

```bibtex
@article{de-marneffe-etal-2021-universal,
    title = "{U}niversal {D}ependencies",
    author = "de Marneffe, Marie-Catherine  and
      Manning, Christopher D.  and
      Nivre, Joakim  and
      Zeman, Daniel",
    journal = "Computational Linguistics",
    volume = "47",
    number = "2",
    month = jun,
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.cl-2.11/",
    doi = "10.1162/coli_a_00402",
    pages = "255--308",
    abstract = "Universal dependencies (UD) is a framework for morphosyntactic annotation of human language, which to date has been used to create treebanks for more than 100 languages. In this article, we outline the linguistic theory of the UD framework, which draws on a long tradition of typologically oriented grammatical theories. Grammatical relations between words are centrally used to explain how predicate{--}argument structures are encoded morphosyntactically in different languages while morphological features and part-of-speech classes give the properties of words. We argue that this theory is a good basis for crosslinguistically consistent annotation of typologically diverse languages in a way that supports computational natural language understanding as well as broader linguistic studies."
}
```

```bibtex
@inproceedings{nivre-etal-2020-universal,
    title = "{U}niversal {D}ependencies v2: An Evergrowing Multilingual Treebank Collection",
    author = "Nivre, Joakim  and
      de Marneffe, Marie-Catherine  and
      Ginter, Filip  and
      Haji{\v{c}}, Jan  and
      Manning, Christopher D.  and
      Pyysalo, Sampo  and
      Schuster, Sebastian  and
      Tyers, Francis  and
      Zeman, Daniel",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.497/",
    pages = "4034--4043",
    language = "eng",
    ISBN = "979-10-95546-34-4",
    abstract = "Universal Dependencies is an open community effort to create cross-linguistically consistent treebank annotation for many languages within a dependency-based lexicalist framework. The annotation consists in a linguistically motivated word segmentation; a morphological layer comprising lemmas, universal part-of-speech tags, and standardized morphological features; and a syntactic layer focusing on syntactic relations between predicates, arguments and modifiers. In this paper, we describe version 2 of the universal guidelines (UD v2), discuss the major changes from UD v1 to UD v2, and give an overview of the currently available treebanks for 90 languages."
}
```

[![arXiv](https://img.shields.io/badge/arXiv-2410.24159-b31b1b.svg)](https://arxiv.org/abs/2410.24159)
[![arXiv](https://img.shields.io/badge/arXiv-2503.10267-b31b1b.svg)](https://arxiv.org/abs/2503.10267)
[![arXiv](https://img.shields.io/badge/arXiv-2511.01066-b31b1b.svg)](https://arxiv.org/abs/2511.01066)

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546].
