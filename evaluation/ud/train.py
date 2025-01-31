import tqdm
import wandb
import argparse
import random
import math
import json
import copy
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from conllu import parse

from transformers import AutoTokenizer

from conll18_ud_eval import evaluate, load_conllu_file
from dataset import Dataset
from model import Model
from lemma_rule import apply_lemma_rule


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)


class CrossEntropySmoothingMasked:
    def __init__(self, smoothing=0.0):
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def __call__(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=1)
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1).clamp(min=0)).squeeze(1)

        logprobs = logprobs.masked_fill(x == float("-inf"), 0.0)
        smooth_loss = -logprobs.sum(dim=1) / (x != float("-inf")).float().sum(dim=1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.masked_fill(target == -1, 0.0).sum() / (target != -1).float().sum()
        return loss


class CollateFunctor:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, sentences):
        longest_source = max([sentence["subwords"].size(0) for sentence in sentences])
        longest_target = max([sentence["upos"].size(0) for sentence in sentences])

        return {
            "index": [sentence["index"] for sentence in sentences],
            "subwords": torch.stack([F.pad(sentence["subwords"], (0, longest_source - sentence["subwords"].size(0)), value=self.pad_index) for sentence in sentences]),
            "alignment": torch.stack(
                [
                    F.pad(F.one_hot(sentence["alignment"], num_classes=longest_target + 2).float(), (0, 0, 0, longest_source - sentence["alignment"].size(0)), value=0.0)
                    for sentence in sentences
                ]
            ),
            "is_unseen": torch.stack([F.pad(sentence["is_unseen"], (0, longest_target - sentence["is_unseen"].size(0)), value=False) for sentence in sentences]),
            "lemma": {
                key: torch.stack([F.pad(sentence["lemma"][key], (0, longest_target - sentence["lemma"][key].size(0)), value=-1) for sentence in sentences])
                for key in sentences[0]["lemma"].keys()
            },
            "upos": torch.stack([F.pad(sentence["upos"], (0, longest_target - sentence["upos"].size(0)), value=-1) for sentence in sentences]),
            "xpos": torch.stack([F.pad(sentence["xpos"], (0, longest_target - sentence["xpos"].size(0)), value=-1) for sentence in sentences]),
            "feats": torch.stack([F.pad(sentence["feats"], (0, longest_target - sentence["feats"].size(0)), value=-1) for sentence in sentences]),
            "arc_head": torch.stack([F.pad(sentence["arc_head"], (0, longest_target - sentence["arc_head"].size(0)), value=-1) for sentence in sentences]),
            "arc_dep": torch.stack([F.pad(sentence["arc_dep"], (0, longest_target - sentence["arc_dep"].size(0)), value=-1) for sentence in sentences]),
            "subword_lengths": torch.LongTensor([sentence["subwords"].size(0) for sentence in sentences]),
            "word_lengths": torch.LongTensor([sentence["upos"].size(0) + 1 for sentence in sentences]),
            "aux_feats_classes": {
                key: torch.stack([F.pad(sentence["aux_feats_classes"][key], (0, longest_target - sentence["aux_feats_classes"][key].size(0)), value=-1) for sentence in sentences])
                for key in sentences[0]["aux_feats_classes"].keys()
            }
        }

def load_data(args, tokenizer):
    language_treebank_mapping = json.load(open(f"language_treebank_mapping_{args.version}.json", "r"))
    treebank = language_treebank_mapping.get(args.language.split('_')[0])
    if treebank is None:
        raise ValueError(f"Treebank not found for {args.language}")
    treebank_path = os.path.join(args.treebank_path, treebank)

    # find train, dev, test filenames
    train_filename, dev_filename, test_filename = None, None, None
    for filename in os.listdir(treebank_path):
        if "train" in filename and ".conllu" in filename:
            train_filename = filename
        elif "dev" in filename and ".conllu" in filename:
            dev_filename = filename
        elif "test" in filename and ".conllu" in filename:
            test_filename = filename
    
    if train_filename is None:
        raise ValueError(f"Train file not found for {args.language}")
    if test_filename is None and dev_filename is not None:
        test_filename = dev_filename
        dev_filename = None
    if test_filename is None:
        raise ValueError(f"Test file not found for {args.language}")

    train_data = Dataset(f"{treebank_path}/{train_filename}", partition='train', tokenizer=tokenizer, add_sep=True, random_mask=True, min_count=args.min_count)
    dev_data = Dataset(f"{treebank_path}/{dev_filename}", partition='dev', tokenizer=tokenizer, forms_vocab=train_data.forms_vocab, lemma_vocab=train_data.lemma_vocab, upos_vocab=train_data.upos_vocab, xpos_vocab=train_data.xpos_vocab, feats_vocab=train_data.feats_vocab, arc_dep_vocab=train_data.arc_dep_vocab, add_sep=True, random_mask=False) if dev_filename is not None else None
    test_data = Dataset(f"{treebank_path}/{test_filename}", partition='test', tokenizer=tokenizer, forms_vocab=train_data.forms_vocab, lemma_vocab=train_data.lemma_vocab, upos_vocab=train_data.upos_vocab, xpos_vocab=train_data.xpos_vocab, feats_vocab=train_data.feats_vocab, arc_dep_vocab=train_data.arc_dep_vocab, add_sep=True, random_mask=False)

    return train_data, dev_data, test_data


def main():
    parser = ArgumentParser()
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default="hplt")
    parser.add_argument("--language", action="store", type=str, default="cs")
    parser.add_argument("--batch_size", action="store", type=int, default=32)
    parser.add_argument("--lr", action="store", type=float, default=0.0005)
    parser.add_argument("--weight_decay", action="store", type=float, default=0.001)
    parser.add_argument("--dropout", action="store", type=float, default=0.3)
    parser.add_argument("--label_smoothing", action="store", type=float, default=0.1)
    parser.add_argument("--epochs", action="store", type=int, default=30)
    parser.add_argument("--seed", action="store", type=int, default=42)
    parser.add_argument("--min_count", action="store", type=int, default=3)
    parser.add_argument("--ema_decay", action="store", type=float, default=0.995)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--treebank_path", default="/scratch/project_465001386/ud-treebanks-v2.13/") # 2.15 for Albanian!
    parser.add_argument("--version", type=str, default="2_0")
    parser.add_argument('--models_path', default='/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/')
    parser.add_argument("--results_path", default="/scratch/project_465001386/hplt-2-0-output/results/")
    parser.add_argument("--checkpoints_path", default="/scratch/project_465001386/hplt-2-0-output/checkpoints/")
    args = parser.parse_args()

    if args.language in ["mr", "ta"]:
        args.batch_size = args.batch_size // 4
        args.lr = args.lr / 2
    elif args.language in ["kk", "ky"]:
        args.batch_size = args.batch_size // 8
        args.lr = 0.0001
        args.dropout = 0.5
        args.epochs = 60

    if args.model == "hplt":
        args.model_path = os.path.join(args.models_path, args.language)
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model {args.model_path} not found")
    elif args.model == "mbert":
        args.model_path = f"bert-base-multilingual-cased"
    elif args.model == "xlmr":
        args.model_path = f"xlm-roberta-base"
    else:
        raise ValueError(f"Unknown model {args.model}")

    seed_everything(args.seed)

    if args.log_wandb:
        wandb.init(name=f"{args.model.split('/')[-1]}_{args.language}", config=args, project="HPLT_UD", entity="ltg", tags=[args.language, args.model])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_data, dev_data, test_data = load_data(args, tokenizer)

    # build and pad with loaders
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=7, collate_fn=CollateFunctor(train_data.pad_index))
    dev_loader = DataLoader(dev_data, args.batch_size, shuffle=False, drop_last=False, num_workers=7, collate_fn=CollateFunctor(train_data.pad_index)) if dev_data is not None else None
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False, drop_last=False, num_workers=7, collate_fn=CollateFunctor(train_data.pad_index))

    model = Model(args, train_data).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.log_wandb:
        wandb.config.update({"params": n_params})
    print(f"{args.language}: {n_params}", flush=True)

    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing).to(device)
    masked_criterion = CrossEntropySmoothingMasked(args.label_smoothing)

    params = list(model.named_parameters())
    no_decay = {'bias', 'LayerNorm', 'vectors', 'embedding', 'layer_score', 'layer_norm'}
    bert_decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay) and "bert" in n]
    bert_no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay) and "bert" in n]
    decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay) and not "bert" in n]
    no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay) and not "bert" in n]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in bert_decay_params], 'lr': 0.1*args.lr, 'weight_decay': 0.1},
        {'params': [p for _, p in bert_no_decay_params], 'lr': 0.1*args.lr, 'weight_decay': 0.0},
        {'params': [p for _, p in decay_params], 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'lr': args.lr, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.99))

    def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = cosine_schedule_with_warmup(optimizer, 250, args.epochs * len(train_loader), 0.1 / 3)

    test_results = {}
    best_mlas_blex_sum = 0.0

    # train loop
    for epoch in range(args.epochs):
        train_iter = tqdm.tqdm(train_loader)
        model.train()
        for batch in train_iter:
            optimizer.zero_grad(set_to_none=True)

            lemma_p, upos_p, xpos_p, feats_p, aux_feats_p, head_p, dep_p, _ = model(
                batch["subwords"].to(device),
                batch["alignment"].to(device),
                batch["subword_lengths"],
                batch["word_lengths"],
                batch["upos"].to(device),
                batch["arc_head"].to(device)
            )

            lemma_loss = {
                key: criterion(p.transpose(1, 2), batch["lemma"][key].to(device))
                for key, p in lemma_p.items()
            }
            lemma_loss = [
                l * (batch["lemma"][key] != -1).float().sum().item() / (batch["feats"] != -1).float().sum().item()
                for key, l in lemma_loss.items()
            ]
            lemma_loss = sum(lemma_loss) / math.sqrt(len(lemma_loss))
            upos_loss = criterion(upos_p.transpose(1, 2), batch["upos"].to(device))
            xpos_loss = criterion(xpos_p.transpose(1, 2), batch["xpos"].to(device))
            feats_loss = criterion(feats_p.transpose(1, 2), batch["feats"].to(device))
            head_loss = masked_criterion(head_p.transpose(1, 2), batch["arc_head"].to(device))
            dep_loss = criterion(dep_p.transpose(1, 2), batch["arc_dep"].to(device))
            aux_feats_loss = {
                key: criterion(p.transpose(1, 2), batch["aux_feats_classes"][key].to(device))
                for key, p in aux_feats_p.items()
            }
            aux_feats_loss = [
                (l * ((batch["aux_feats_classes"][key] != -1).float().sum().item() / (batch["feats"] != -1).float().sum().item())) if (batch["aux_feats_classes"][key] != -1).float().sum().item() > 0 else 0.0
                for key, l in aux_feats_loss.items()
            ]
            aux_feats_loss = (sum(aux_feats_loss) / len(aux_feats_loss)) if len(aux_feats_loss) > 0 else 0.0

            loss = lemma_loss + upos_loss + xpos_loss + feats_loss + head_loss + dep_loss + aux_feats_loss
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
                    param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

            if args.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/lemma_loss": lemma_loss.item(),
                        "train/upos_loss": upos_loss.item(),
                        "train/xpos_loss": xpos_loss.item(),
                        "train/feats_loss": feats_loss.item(),
                        "train/head_loss": head_loss.item(),
                        "train/dep_loss": dep_loss.item(),
                        "train/aux_feats_loss": aux_feats_loss.item() if not isinstance(aux_feats_loss, float) else 0.0,
                        "train/loss": loss.item(),
                        "stats/grad_norm": grad_norm.item(),
                        "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    }
                )
            train_iter.set_postfix_str(f"loss: {loss.item()}")

        # save checkpoint
        torch.save({
                "model": ema_model.state_dict(),
                "dataset": train_data.state_dict()
            },
            f"{args.checkpoints_path}ud-{args.language}-{args.model}.bin"
        )

        # eval
        for loader_index, (loader, dataset) in enumerate([(dev_loader, dev_data), (test_loader, test_data)]):
            if loader is None:
                continue

            with torch.no_grad():
                ema_model.eval()

                dev_file = parse(open(dataset.path, "r").read())
                prediction_folder = os.path.join(args.results_path, 'tmp')
                if not os.path.exists(prediction_folder):
                    os.makedirs(prediction_folder)
                prediction_path = f"{prediction_folder}/{args.language}_{args.model}_{loader_index}.conllu"
                with open(prediction_path, "w") as f:
                    for batch in loader:
                        lemma_p, upos_p, xpos_p, feats_p, _, __, dep_p, head_p = ema_model(
                            batch["subwords"].to(device),
                            batch["alignment"].to(device),
                            batch["subword_lengths"],
                            batch["word_lengths"],
                        )

                        for i, index in enumerate(batch["index"]):
                            offset = 0
                            for j, form in enumerate(dataset.forms[index]):
                                # if combined, skip
                                while not type(dev_file[index][j + offset]["id"]) is int:
                                    offset += 1
                                new_j = j + offset

                                lemma_rule = {
                                    rule_type: dataset.lemma_vocab[rule_type].get(p[i, j, :].argmax().item(), dataset.lemma_vocab[rule_type][-1])
                                    for rule_type, p in lemma_p.items()
                                }
                                dev_file[index][new_j]["lemma"] = apply_lemma_rule(form, lemma_rule)
                                dev_file[index][new_j]["upos"] = dataset.upos_vocab[upos_p[i, j, :].argmax().item()]
                                dev_file[index][new_j]["xpos"] = dataset.xpos_vocab[xpos_p[i, j, :].argmax().item()]
                                dev_file[index][new_j]["feats"] = dataset.feats_vocab[feats_p[i, j, :].argmax().item()]
                                dev_file[index][new_j]["head"] = head_p[i, j].item()
                                dev_file[index][new_j]["deprel"] = dataset.arc_dep_vocab[dep_p[i, j, :].argmax().item()]
                                dev_file[index][new_j]["deps"] = "_" 

                            f.write(dev_file[index].serialize())

            try:
                gold_ud = load_conllu_file(dataset.path)
                system_ud = load_conllu_file(prediction_path)
                evaluation = evaluate(gold_ud, system_ud)
            except:
                break

            if args.log_wandb and loader_index == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        f"valid/dev_UPOS": evaluation["UPOS"].aligned_accuracy * 100,
                        f"valid/dev_XPOS": evaluation["XPOS"].aligned_accuracy * 100,
                        f"valid/dev_UFeats": evaluation["UFeats"].aligned_accuracy * 100,
                        f"valid/dev_AllTags": evaluation["AllTags"].aligned_accuracy * 100,
                        f"valid/dev_Lemmas": evaluation["Lemmas"].aligned_accuracy * 100,
                        f"valid/dev_UAS": evaluation["UAS"].aligned_accuracy * 100,
                        f"valid/dev_LAS": evaluation["LAS"].aligned_accuracy * 100,
                        f"valid/dev_MLAS": evaluation["MLAS"].aligned_accuracy * 100,
                        f"valid/dev_CLAS": evaluation["CLAS"].aligned_accuracy * 100,
                        f"valid/dev_BLEX": evaluation["BLEX"].aligned_accuracy * 100
                    }
                )

            if loader_index == 0:
                print(evaluation["MLAS"].aligned_accuracy * 100, evaluation["BLEX"].aligned_accuracy * 100, flush=True)
                mlas_blex_sum = evaluation["MLAS"].aligned_accuracy + evaluation["BLEX"].aligned_accuracy
                if mlas_blex_sum < best_mlas_blex_sum:
                    break
                best_mlas_blex_sum = mlas_blex_sum

            else:
                results = {
                    "UPOS": evaluation["UPOS"].aligned_accuracy * 100,
                    "XPOS": evaluation["XPOS"].aligned_accuracy * 100,
                    "UFeats": evaluation["UFeats"].aligned_accuracy * 100,
                    "AllTags": evaluation["AllTags"].aligned_accuracy * 100,
                    "Lemmas": evaluation["Lemmas"].aligned_accuracy * 100,
                    "UAS": evaluation["UAS"].aligned_accuracy * 100,
                    "LAS": evaluation["LAS"].aligned_accuracy * 100,
                    "CLAS": evaluation["CLAS"].aligned_accuracy * 100,
                    "MLAS": evaluation["MLAS"].aligned_accuracy * 100,
                    "BLEX": evaluation["BLEX"].aligned_accuracy * 100
                }

                # save results; lock and rewrite results.json
                test_results[f"{args.language}_{args.model}"] = results
                with open(f"{args.results_path}{args.language}_{args.model}.jsonl", "w") as f:
                    json.dump(test_results, f)
                

if __name__ == '__main__':
    main()
