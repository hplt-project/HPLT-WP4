# HPLT-WP4

This repository contains information and code realted to HPLT Work Package 4: High Performance Language Models

## Objectives

This work package will train and evaluate language models, both mono- and multi-lingual. Comprehensive eval uation will use both intrinsic and extrinsic measures of model quality. The work package additionally emphasizes the efficiency of data and computational resources, exploring e.g. various alternative training objectives, as well as ethical considerations in model training and use, in particular approaches for quantifying and reducing model biases internalized from training data.

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
