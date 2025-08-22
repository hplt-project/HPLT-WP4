import argparse
import os
import random

from datasets import load_dataset
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import pytorch_lightning as pl

from fine_tuner import T5FineTuner
from logging_callback import LoggingCallback
from ner_dataset import WikiAnnDataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/scratch/project_465001925/mariiaf/swh-Latn-t5')
    parser.add_argument('--lang', default='sw')
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--output_dir', default="/scratch/project_465001890/hplt-2-0-output/t5-ner")
    return parser.parse_args()

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
set_seed(42)

torch.set_float32_matmul_precision('high')
cl_args = parse_arguments()
tokenizer = AutoTokenizer.from_pretrained(
            cl_args.model_path,
        )
checkpoint_path = cl_args.output_dir+"/checkpoint.pth"
dataset = load_dataset("wikiann", cl_args.lang)
if cl_args.do_train:
    args_dict = dict(
      data_dir="wikiann", # path for data files
      output_dir=cl_args.output_dir, # path to save the checkpoints
      model_path=cl_args.model_path,
      max_seq_length=512,
      learning_rate=3e-4,
      weight_decay=0.0,
      adam_epsilon=1e-8,
      warmup_steps=0,
      train_batch_size=20,
      eval_batch_size=32,
      num_train_epochs=1,
      devices=1,
      gradient_accumulation_steps=1,
      early_stop_callback=False,
      fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
      opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
      max_grad_norm=1, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
      seed=42,
    )
    args = argparse.Namespace(**args_dict)
    model = T5FineTuner(args, tokenizer, dataset)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
      filename=checkpoint_path, monitor="val_loss", mode="min", save_top_k=5
    )


    train_params = dict(
      accumulate_grad_batches=args.gradient_accumulation_steps,
      devices=args.devices,
      accelerator="gpu",
      max_epochs=args.num_train_epochs,
      #early_stop_callback=False,
      precision= 16 if args.fp_16 else 32,
      #amp_level=args.opt_level,
      callbacks=[LoggingCallback(), checkpoint_callback],
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

model = T5FineTuner.load_from_checkpoint(checkpoint_path + '.ckpt')
test_dataset = WikiAnnDataset(tokenizer=tokenizer, dataset=dataset, type_path='test')
test_loader = DataLoader(test_dataset, batch_size=32,
                             num_workers=2, shuffle=True)
model.model.eval()
model = model.to("cuda")
outputs = []
targets = []
all_text = []
true_labels = []
pred_labels = []

def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind+sll-1))
    return results
    
def generate_label(input: str, target: str):
    mapper = {'O': 0, 'B-DATE': 1, 'I-DATE': 2, 'B-PER': 3,
              'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}
    inv_mapper = {v: k for k, v in mapper.items()}

    input = input.split(" ")
    target = target.split("; ")

    init_target_label = [mapper['O']]*len(input)

    for ent in target:
        ent = ent.split(": ")
        try:
            sent_end = ent[1].split(" ")
            index = find_sub_list(sent_end, input)
        except:
            continue
        # print(index)
        try:
            init_target_label[index[0][0]] = mapper[f"B-{ent[0].upper()}"]
            for i in range(index[0][0]+1, index[0][1]+1):
                init_target_label[i] = mapper[f"I-{ent[0].upper()}"]
        except:
            continue
    init_target_label = [inv_mapper[j] for j in init_target_label]
    return init_target_label

for batch in tqdm(test_loader):
    input_ids = batch['source_ids'].to("cuda")
    attention_mask = batch['source_mask'].to("cuda")
    outs = model.model.generate(input_ids=input_ids,
                                attention_mask=attention_mask)
    dec = [tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip() for ids in outs]
    target = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                for ids in batch["target_ids"]]
    texts = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                for ids in batch["source_ids"]]
    true_label = [generate_label(texts[i].strip(), target[i].strip()) if target[i].strip() != 'none' else [
        "O"]*len(texts[i].strip().split()) for i in range(len(texts))]
    pred_label = [generate_label(texts[i].strip(), dec[i].strip()) if dec[i].strip() != 'none' else [
        "O"]*len(texts[i].strip().split()) for i in range(len(texts))]

    outputs.extend(dec)
    targets.extend(target)
    true_labels.extend(true_label)
    pred_labels.extend(pred_label)
    all_text.extend(texts)

metric = evaluate.load("seqeval")

for i in range(10):
    print(f"Text:  {all_text[i]}", flush=True)
    print(f"Predicted Token Class:  {pred_labels[i]}", flush=True)
    print(f"True Token Class:  {true_labels[i]}", flush=True)
    print("=====================================================================\n", flush=True)

print(metric.compute(predictions=pred_labels, references=true_labels), flush=True)