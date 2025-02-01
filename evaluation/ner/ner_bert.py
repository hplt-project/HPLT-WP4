"""
A simplified version of the run_ner.py script from the transformers library,
https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification
The script does sequence labeling, aka token classification, on datasets with BIO tags.
Preferred usage is to pass a json file with configuration parameters as the only argument.
Alternatively,  pass individual settings as keyword arguments.
In default_params, the arguments that can be passed are defined.
Any ModelArguments, DataTrainingArguments,
TrainingArguments may be defined here or in the json file.

For each task, specify path to the dataset in "dataset_name", specify "task_name", "output_dir"
and "label_column_name" as well.
label_column_name is "tsa_tags" in the TSA dataset and  "ner_tags" in the NER dataset.
"""

import argparse
import json
import os
from pathlib import Path
import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
from numpyencoder import NumpyEncoder
from constants import LANGS_MAPPING
from tsa_utils import ModelArguments, DataTrainingArguments
from tsa_utils import tsa_eval

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
print("Numpy:", np.version.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

hf_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
# default_params define what named parameters can be sent to the script
default_params = {  # Add entries here that you want to change or just track the value of
    "model_name_or_path": "ltg/norbert3-small",
    "trust_remote_code": True,
    "dataset_name": "wikiann/",
    "max_seq_length": 256,
    "seed": 101,
    "per_device_train_batch_size": 64,
    "task_name": "ner",
    "output_dir": "ner_results",
    "overwrite_cache": True,
    "overwrite_output_dir": True,
    "do_train": True,
    "num_train_epochs": 5,
    "do_eval": True,
    "return_entity_level_metrics": True,
    # Since we use a separate evaluation script, this is better kept False
    "use_auth_token": False,
    "logging_strategy": "epoch",  # "epoch"
    "save_strategy": "epoch",  # "epoch"
    "evaluation_strategy": "epoch",  # "epoch"
    "save_total_limit": 2,
    "load_best_model_at_end": True,  # Evaluate the last epoch
    "label_column_name": "ner_tags",
    "disable_tqdm": False,
    "do_predict": True,
    "text_column_name": "tokens",
}

parser = argparse.ArgumentParser(
    description="Pass the path to a json file with configuration parameters "
                "as positional argument, or pass individual settings as keyword arguments."
)
parser.add_argument("config", nargs="?")
for key, value in default_params.items():
    parser.add_argument(f"--{key}", default=value, type=type(value))
args = parser.parse_args()
if args.config is not None:
    with open(os.path.abspath(args.config)) as rf:
        args_dict = json.load(rf)
else:
    args_dict = vars(args)

args_dict.pop("config", None)
# Since we are flexible with what arguments are defined, we need to convert

label_col = args_dict["label_column_name"]


def compute_metrics(p):
    cur_predictions, cur_labels = p
    cur_predictions = np.argmax(cur_predictions, axis=2)

    # Remove ignored index (special tokens)
    cur_true_predictions = [
        [labelnames[p] for (p, el) in zip(prediction, label) if el not in {-100, -101}]
        for prediction, label in zip(cur_predictions, cur_labels)
    ]
    true_labels = [
        [labelnames[el] for (p, el) in zip(prediction, label) if el not in {-100, -101}]
        for prediction, label in zip(cur_predictions, cur_labels)
    ]

    results = metric.compute(predictions=cur_true_predictions, references=true_labels)
    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for cur_key, cur_value in results.items():
            if isinstance(cur_value, dict):
                for n, v in cur_value.items():
                    final_results[f"{cur_key}_{n}"] = v
            else:
                final_results[cur_key] = cur_value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def get_label_list(cur_labels):
    # ner_tags: a list of classification labels, with possible values including
    # O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4), B-LOC (5), I-LOC (6).
    unique_labels = set()
    for label in cur_labels:
        unique_labels = unique_labels | set(label)
    cur_label_list = list(unique_labels)
    # sorted_labels = sorted(
    #    cur_label_list, key=lambda name: (name[1:], name[0])
    # )  # Gather B and I
    return cur_label_list


# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # We use this argument because the texts in our dataset are lists of words
        # (with a label for each word).
        is_split_into_words=True,
    )
    cur_labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None.
            # We set the label to -100, so they are automatically
            # ignored in the loss function.
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-101)
            # We set the label for the first token of each word only.
            else:  # New word
                label_ids.append(label_to_id[label[word_idx]])
            # We do not keep the option to label the subsequent subword tokens here.

            previous_word_idx = word_idx

        cur_labels.append(label_ids)
    tokenized_inputs["labels"] = cur_labels
    return tokenized_inputs


model_args, data_args, training_args = hf_parser.parse_dict(args_dict)

text_column_name = data_args.text_column_name
label_column_name = data_args.label_column_name
assert data_args.label_all_tokens is False, "Our script only labels first subword token"
ds, lang = data_args.dataset_name.split("/")
if len(lang) > 2:
    lang = LANGS_MAPPING[lang.split("_")[0]]
print(f"loading {ds} for lang {lang}")
dsd = load_dataset(ds, lang)
transformers.logging.set_verbosity_warning()


label_list = get_label_list(dsd["train"][data_args.label_column_name])
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)
labels_are_int = True

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    trust_remote_code=model_args.trust_remote_code,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
tokenizer_name_or_path = (
    model_args.tokenizer_name
    if model_args.tokenizer_name
    else model_args.model_name_or_path
)
if config.model_type in {"gpt2", "roberta"}:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

# %%
# Instanciate the model
model = AutoModelForTokenClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    trust_remote_code=model_args.trust_remote_code,
    revision=model_args.model_revision,
    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
)

print(f"Our label2id: {label_to_id}")

assert (model.config.label2id == PretrainedConfig(num_labels=num_labels).label2id) or (
    model.config.label2id == label_to_id
), "Model seems to have been fine-tuned on other labels already. Our script does not adapt to that."

# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {i: l for i, l in enumerate(label_list)}

# Preprocessing the dataset
# Padding strategy
padding = "max_length" if data_args.pad_to_max_length else False

with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = dsd["train"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    eval_dataset = dsd["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    predict_dataset = dsd["test"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on test dataset",
    )

print("Dataset features are now:", list(train_dataset.features))

data_collator = DataCollatorForTokenClassification(
    tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
)

# Metrics
metric = evaluate.load("seqeval")  #
labelnames = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train(resume_from_checkpoint=False)
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Evaluate
print("\nEvaluation,", model_args.model_name_or_path)


trainer_predict = trainer.predict(predict_dataset, metric_key_prefix="predict")
predictions, labels, metrics = trainer_predict

predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [labelnames[p] for (p, el) in zip(prediction, label) if el != -101]
    for prediction, label in zip(predictions, labels)
]

gold = predict_dataset[
    args_dict["label_column_name"]
]  # Note: Will not work if dataset has ints, and the text labels in metadata
gold = [[labelnames[p] for p in sent] for sent in gold]

for g, pred in zip(gold, true_predictions):
    try:
        assert (len(g) == len(pred))
    except AssertionError:
        print((len(g), len(pred)))
        print(g)
        print(pred)
        raise AssertionError


batista_f1 = tsa_eval(gold, true_predictions)
print("batista_f1", batista_f1)

seqeval_results = metric.compute(predictions=true_predictions, references=gold)
print(seqeval_results)

args_dict["test_f1"] = batista_f1
args_dict["seqeval"] = seqeval_results

modelname = args_dict["model_name_or_path"]
datasetname = args_dict["dataset_name"]
score = args_dict["seqeval"]["overall_f1"]

with open("ner_results.tsv", "a") as myfile:
    myfile.write(f"{modelname}\t{datasetname}\t{score:.3f}\n")

save_path = Path(args_dict["output_dir"]).resolve()
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(save_path, args_dict["task_name"] + "_results.json").write_text(
    json.dumps(args_dict, cls=NumpyEncoder)
)
