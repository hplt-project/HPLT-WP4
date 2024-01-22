# coding=utf-8

import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname
from tokenizers import Tokenizer
from statistics import mean
import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from lamb import Lamb
from config import BertConfig
from model import Bert
from utils import cosine_schedule_with_warmup_cooldown, is_main_process, get_rank, seed_everything, get_world_size
from dataset import Dataset, ValidationDataset, apply_mask


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--language", default="en", type=str, help="The language to train on.")
    parser.add_argument("--input_dir", default="/scratch/project_465000498/processed_data/{language}", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--name", default="bert_base_{language}", type=str)
    parser.add_argument("--config_file", default="configs/base.json", type=str, help="The BERT model config")
    parser.add_argument("--output_dir", default="/scratch/project_465000498/hplt_models", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to a previous checkpointed training state.")
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--seq_length", default=128, help="Sequence length for training.")
    parser.add_argument("--batch_size", default=256, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=31250, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--validate_every", default=1, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=31250//10, help="save every X steps")
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging loss.')
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Masking probability.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Masking probability.")
    parser.add_argument("--short_p", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--optimizer_eps", default=1e-6, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    args.input_dir = args.input_dir.format(language=args.language)
    args.name = args.name.format(language=args.language)
    args.tokenizer_path = f"{args.input_dir}/tokenizer.json"
    args.output_dir = f"{args.output_dir}/{args.name}"

    return args


@torch.no_grad()
def log_parameter_histograms(model, step):
    for name, param in model.named_parameters():
        wandb.log(
            {
                f"parameters/norm_{name}": torch.linalg.norm(param.data).cpu().item(),
                f"parameters/std_{name}": param.data.std().cpu().item(),
            },
            step=step,
            commit=False
        )
        if param.requires_grad and param.grad is not None:
            wandb.log(
                {
                    f"gradients/norm_{name}": torch.linalg.norm(param.grad).cpu().item(),
                    f"gradients/std_{name}": param.grad.std().cpu().item(),
                },
                step=step,
                commit=False
            )


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    seed_everything(args.seed + rank)

    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"RCCL started on device {device}", flush=True)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if is_main_process():
        os.system(f"mkdir -p {args.output_dir}")

    args.n_training_files = len(fnmatch.filter(os.listdir(f"{args.input_dir}/tokenized_shards"), "train_*.pt.gz"))

    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")
        print(f"Found {args.n_training_files} training shards", flush=True)

    args.mask_token_id = tokenizer.token_to_id("[MASK]")
    args.cls_token_id = tokenizer.token_to_id("[CLS]")
    args.pad_token_id = tokenizer.token_to_id("[PAD]")
    args.sep_token_id = tokenizer.token_to_id("[SEP]")
    args.vocab_size = tokenizer.get_vocab_size()
    args.n_special_tokens = tokenizer.token_to_id("[MASK_99]") + 2

    if is_main_process():
        wandb.init(
            name=args.name,
            config=args,
            id=args.wandb_id,
            project="HPLT",
            entity="ltg",
            resume="auto",
            allow_val_change=True,
            reinit=True
        )

    return device, local_rank


def prepare_model_and_optimizer(args, device, local_rank, checkpoint):
    config = BertConfig(args.config_file)
    config.vocab_size = args.vocab_size
    model = Bert(config)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(config.to_dict())
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)

    no_decay = ['bias', 'layer_norm', '_embedding']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
 
    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1
    )

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    return model, config, optimizer, scheduler


def training_epoch(model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, args, device, max_local_steps):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    if is_main_process():
        train_iter = tqdm(train_dataloader, desc="Train iteration", initial=global_step, total=args.max_steps)
    else:
        train_iter = train_dataloader

    for local_step, batch in enumerate(train_iter):
        input_ids, attention_mask, mask_ratios, replacement_tokens = [t.to(device, non_blocking=True) for t in batch]
        input_ids, mask_ratios, replacement_tokens = input_ids.t(), mask_ratios.t(), replacement_tokens.t()
        input_ids, target_ids, mask_p = apply_mask(args, input_ids, mask_ratios, replacement_tokens, global_step)

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, perplexity, accuracy = model(input_ids, attention_mask, target_ids)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

        optimizer.step()

        scheduler.step()
        global_step += 1

        with torch.no_grad():
            metrics = torch.stack([loss, perplexity, accuracy])
            torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
            loss, perplexity, accuracy = metrics.tolist()

        if is_main_process():
            train_iter.set_postfix_str(f"loss: {loss:.2f}, accuracy: {accuracy * 100.0:.2f}, grad_norm: {grad_norm:.2f}, lr: {optimizer.param_groups[0]['lr']:.5f}")

            if global_step % 100 == 0:
                log_parameter_histograms(model, global_step)

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": loss,
                    "train/perplexity": perplexity,
                    "train/accuracy": accuracy * 100.0,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": grad_norm,
                    "stats/seq_length": args.seq_length,
                    "stats/mask_p": mask_p,
                }
            )

        optimizer.zero_grad(set_to_none=True)

        if global_step % args.save_every == 0:
            save(model, optimizer, scheduler, global_step, epoch, args)
            validation_epoch(model, valid_dataloader, epoch, args, device)
            model = model.train()

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps or local_step >= max_local_steps - 1:
            return global_step
        if global_step == (args.max_steps // 10 * 7) or global_step == (args.max_steps // 10 * 9):
            return global_step

    return global_step


@torch.no_grad()
def validation_epoch(model, valid_dataloader, epoch, args, device):
    model = model.eval()

    if is_main_process():
        valid_iter = tqdm(valid_dataloader, desc="Validation iteration")
    else:
        valid_iter = valid_dataloader

    losses, perplexities, accuracies = [], [], []
    for batch in valid_iter:
        input_ids, attention_mask, mask_ratios, replacement_tokens = [t.to(device, non_blocking=True) for t in batch]
        input_ids, mask_ratios, replacement_tokens = input_ids.t(), mask_ratios.t(), replacement_tokens.t()
        input_ids, target_ids, _ = apply_mask(args, input_ids, mask_ratios, replacement_tokens, args.max_steps)

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, perplexity, accuracy = model(input_ids, attention_mask, target_ids)

        metrics = torch.stack([loss, perplexity, accuracy])
        torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        loss, perplexity, accuracy = metrics.tolist()

        losses.append(loss)
        perplexities.append(perplexity)
        accuracies.append(accuracy)

    if is_main_process():
        wandb.log(
            {
                "epoch": epoch,
                "validation/loss": mean(losses),
                "validation/accuracy": mean(accuracies) * 100.0,
                "validation/perplexity": mean(perplexities)
            }
        )


def save(model, optimizer, scheduler, global_step, epoch, args):
    checkpoint_path = f"{args.output_dir}/model_step_{global_step}.bin"
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path
        )


def load_datasets(args, tokenizer, epoch, global_step, device, train_data, valid_data):
    train_index = (get_rank() + epoch * get_world_size()) % args.n_training_files
    train_seed = args.seed + get_rank() + epoch * get_world_size()
    train_path = f"{args.input_dir}/tokenized_shards/train_{train_index:05d}.pt.gz"

    if (global_step + 1) / args.max_steps >= 0.9:
        args.seq_length = 512
        batch_size = args.batch_size // 4
    elif (global_step + 1) / args.max_steps >= 0.7:
        args.seq_length = 256
        batch_size = args.batch_size // 2
    else:
        args.seq_length = 128
        batch_size = args.batch_size

    if train_data is None or train_data.path != train_path or train_data.seq_length != args.seq_length:
        train_data = Dataset(train_path, tokenizer, args)
        print(f"Loaded training file {train_index} on GPU {get_rank()}", flush=True)
        if is_main_process():
            train_data.show_random_item(tokenizer)

    if valid_data is None:
        valid_data = ValidationDataset(f"{args.input_dir}/tokenized_shards/validation.pt.gz", tokenizer, int(os.environ["SLURM_PROCID"]), int(os.environ["WORLD_SIZE"]), args)

    min_length = torch.tensor(len(train_data) // batch_size // 2, dtype=torch.long, device=device)
    torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,  # non-zero num_workers cause segmenation fault
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_data,
        shuffle=False,
        batch_size=256,
        num_workers=0,  # non-zero num_workers cause segmenation fault
        generator=torch.Generator().manual_seed(42),
        drop_last=True,
        pin_memory=True,
    )

    return train_data, valid_data, train_dataloader, valid_dataloader, min_length


if __name__ == "__main__":
    args = parse_arguments()

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        checkpoint_args, initial_epoch, global_step = checkpoint["args"], checkpoint["epoch"] + 1, checkpoint["global_step"]
        args = vars(args).copy()
        args.update(vars(checkpoint_args))
        args = argparse.Namespace(**args)
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0
        args.wandb_id = wandb.util.generate_id() if int(os.environ["SLURM_PROCID"]) == 0 else 0

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    device, local_rank = setup_training(args, tokenizer)
    model, config, optimizer, scheduler = prepare_model_and_optimizer(args, device, local_rank, checkpoint)
    train_data, valid_data = None, None

    for epoch in count(initial_epoch):
        train_data, valid_data, train_dataloader, valid_dataloader, min_length = load_datasets(args, tokenizer, epoch, global_step, device, train_data, valid_data)
        global_step = training_epoch(model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, args, device, min_length)

        if global_step >= args.max_steps:
            break

    save(model, optimizer, scheduler, global_step, epoch, args)
    validation_epoch(model, valid_dataloader, epoch, args, device)
