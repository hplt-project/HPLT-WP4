# HPLT-WP4

This repository contains information and code realted to HPLT Work Package 4: High Performance Language Models

## Objectives

This work package will train and evaluate language models, both mono- and multi-lingual. Comprehensive evaluation will use both intrinsic and extrinsic measures of model quality. The work package additionally emphasizes the efficiency of data and computational resources, exploring e.g. various alternative training objectives, as well as ethical considerations in model training and use, in particular approaches for quantifying and reducing model biases internalized from training data.

Related Milestones: MS2, MS4, MS5, MS6.

## Subtasks

### Task T4.1: Building/Training Language Models (UTURKU, UOSLO) 

We will adapt and further develop existing tools for training bidirectional language models for dozens of languages (e.g. Pyysalo et al., 2021; Ravishankar et al., 2021) and build on tools currently in development at UTURKU and UOSLO for training causal and encoder-decoder language models on the LUMI supercomputer to create a fully automated, unified and comprehensively documented process for LM training. The developed tools will be made openly available throughout development to facilitate replicability and collaboration with related efforts. Using data sourced in WP2 and cleaned and preprocessed in WP3, these tools will be applied to create causal, bidirectional and encoder-decoder models for all targeted languages, building both state-of-the-art monolingual models for each language as well as various multilingual models. The trained models will be stored and released in a fully systematic, documented way, following the path of the NLPL Word Embeddings Repository (Fares et al., 2017) maintained by UOSLO and also made available through community repositories such as that maintained by HuggingFace along with detailed documentation following community best practices such as model cards (Mitchell et al., 2019).

### Task T4.2: Efficient Data Usage & HPC utilization (UOSLO)

We will prioritize research of more efficient and accessible language models, and devise a careful curation of training data. Current language models are trained on naive self-supervised tasks, which carry weak training signal – these tasks can be mostly resolved
locally, which does not force the models to build a rich language representation. We will explore and benchmark other self-supervision tasks; we argue this space is highly under-explored in current language modeling. Plenty of promising ideas tackling this apparent issue have surged in the last years. For example, Xu et al. (2021) augment the mask reconstruction by predicting distances in the underlying dependency trees, while Joshi et al. (2020) and Levine et al. (2021) train more efficient models by masking multi-word spans instead of random sub-tokens. In ELECTRA, Clark et al. (2020) reformulate language modeling as a min-max game between two models to achieve faster training. We will build upon the existing methods and test more linguistically-motivated approaches, which utilize weak supervision signals from the enormous sources of high-quality hand-annotated corpora. From a more bottom-up perspective, it is also important to carefully choose the right training data for self-supervision – for example, CamemBERT (Martin et al., 2020) performs similarly when trained on a full 138GB corpus and when trained just on a 4GB subset of it. We will make sure the training data are cautiously selected to not waste resources. Availability of internally structured monolingual training data, paired with metadata and the previously unmatched space for systematic experimentation created on LUMI will allow distilling relevant best practices and making available relevant techniques for broader re-use.

### Task T4.3: Evaluating Large Language Models (UTURKU, UOSLO)

All produced language models will be evaluated using standard intrinsic metrics of language model quality, such as the perplexity of the model on held-out test data. Additionally, we will assemble resources and create tools for assessing model quality through
their performance as part of systems performing various downstream natural language processing tasks, such as syntactic analysis and named entity recognition. We will assemble a broad selection of task-specific multilingual datasets such as Universal Dependencies (Nivre et al., 2020), WikiANN (Rahimi et al., 2019), and XED (Öhman et al., 2020), and develop a framework for model evaluation at each task. This framework will be applied to assess language models produced in the project with respect to other such models, e.g. to identify languages with potential issues in the data or the pre-training process, as well as with respect to other available models (e.g. massively multilingual models) to confirm that the pre-training process produces models that support state-of-the-art downstream applications.

### Task T4.4: Ethical Considerations in Training and Deployment (UOSLO, CUNI)

This task implements our consolidated ethics for language modeling, with a focus on exploring debiasing in an end-to-end fashion that was previously too costly to try.

## Deliverables

- **D4.1 OTHER/PU M30 UTURKU Trained language models (software)**
- **D4.2 R/PU M35 UTURKU Report on language model evaluation**


## How to run

Run `schedule.sh` to schedule jobs for creating shards, training a tokenizer, tokenizing shards and training BERT.

For any training, don't forget to make sure that the tokenized shards are all of equal size (see below).

### BERTs

```
sbatch schedule.sh \
    nno_Latn \
    /appl/local/openeurollm/training/catalogue/hplt/3.0/sorted/nno_Latn \
    <output directory> \
    512 \
    0.0 \
    --run_training
```

### T5s

```
sbatch schedule.sh \
    nno_Latn \
    /appl/local/openeurollm/training/catalogue/hplt/3.0/sorted/nno_Latn \
    /scratch/project_465002259/hplt-3-0-t5/nno_Latn \
    512 \
    0.0
```

and then follow instructions in [encoder-decoder/README.md](encoder-decoder/README.md). 

### Edge cases

#### Large languages: > 200M documents

For languages like Dutch, French, Indonesian, Japanese etc. we have more documents than it is possible to process in 31250 global steps.
So we don't use all of them, but randomly sample from all *.zst files (an experiment with an English BERT has shown having diverse data works better than only those scored WSD 10).
This sampling is done by the command

```
sbatch schedule.sh \
    nld_Latn \
    /appl/local/openeurollm/training/catalogue/hplt/3.0/sorted/nld_Latn \
    /scratch/project_465002259/hplt-3-0-t5/nld_Latn \
    512 \
    0.0 \
    --n_training_documents 20000000
```

20000000 is the number of documents that was enough for 31250 global steps for the languages we tried.

#### Extremely large languages: > 1B documents

For languages like English and Russian, if sharding "as is", we might submit more jobs than allowed by LUMI. It might result with too few data for training, too many epochs and overfitting. Sharding of these languages is done by the command

```
sbatch schedule.sh \
    eng_Latn \
    /appl/local/openeurollm/training/catalogue/hplt/3.0/sorted/eng_Latn \
    /scratch/project_465002259/hplt-3-0-t5/eng_Latn \
    512 \
    0.0 \
    --n_training_documents 20000000 \
    --max_n_jobs 79
```
!! max_n_jobs must be chosen from how many jobs on the LUMI's `small` you already run and from the number of *.zst files for the language. The number of *.zst files must be as much as possible divisible by max_n_jobs (the remainder will be appended to the first batch and if it is large, it will be a very long running time).

### Inputs for training must be of equal size

Sampling of documents in large languages results in tokenized shards of very different sizes (because we don't know how many tokens a document contains before tokenizing it, and documents from *.zst scored 10 tend to be longer than scored e.g. 5). 

Manually rename the folder to make sure you don't overwrite something that had been processed for hours and don't forget to copy back the validation file after it has worked as it was supposed to

```
mv /scratch/project_465002259/hplt-3-0-t5/nld_Latn/tokenized_shards/ /scratch/project_465002259/hplt-3-0-t5/nld_Latn/tokenized_neq_shards
cd utils
sbatch rewrite.sh --lang nld_Latn 
cp /scratch/project_465002259/hplt-3-0-t5/nld_Latn/tokenized_neq_shards/validation.pt.gz /scratch/project_465002259/hplt-3-0-t5/nld_Latn/tokenized_shards/validation.pt.gz
```

## Cite

### HPLT 3.0 paper

```
@misc{oepen2025hplt30largescalemultilingual,
      title={HPLT 3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models}, 
      author={Stephan Oepen and Nikolay Arefev and Mikko Aulamo and Marta Bañón and Maja Buljan and Laurie Burchell and Lucas Charpentier and Pinzhen Chen and Mariia Fedorova and Ona de Gibert and Barry Haddow and Jan Hajič and Jindřich Helcl and Andrey Kutuzov and Veronika Laippala and Zihao Li and Risto Luukkonen and Bhavitvya Malik and Vladislav Mikhailov and Amanda Myntti and Dayyán O'Brien and Lucie Poláková and Sampo Pyysalo and Gema Ramírez Sánchez and Janine Siewert and Pavel Stepachev and Jörg Tiedemann and Teemu Vahtola and Dušan Variš and Fedor Vitiugin and Tea Vojtěchová and Jaume Zaragoza},
      year={2025},
      eprint={2511.01066},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01066}, 
}
```

### HPLT 2.0 paper

```
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

### T5 paper

```
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

### BERT paper

```
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
