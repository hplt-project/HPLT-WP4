# coding=utf-8

import os
import os.path
import argparse
import fnmatch
from tqdm import tqdm
from datetime import timedelta
from itertools import count
from socket import gethostname
from statistics import mean
from tokenizers import Tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from lamb import Lamb
from config import BertConfig
from model import T5
from utils import is_main_process, get_rank, seed_everything, get_world_size, cosine_schedule_with_warmup
from dataset import Dataset, CollateFunctor


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir", default="../bert-lumi/data/pretrain/tokenized", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--name", default="base", type=str)
    parser.add_argument("--config_file", default="configs/base.json", type=str, help="The BERT model config")
    parser.add_argument("--output_dir", default="checkpoints/base", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--tokenizer_path", default="../bert-lumi/data/wordpiece.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to a previous checkpointed training state.")

    # Other parameters
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--seq_length", default=512, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--validate_every", default=1, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=2e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=250000, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.004, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging loss.')
    parser.add_argument("--mask_p", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--short_p", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--data_format", default="pt.gz")
    parser.add_argument('--save_every', type=int, default=3125, help="save every X steps")
    args = parser.parse_args()

    return args


def setup_training(args):
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

    args.n_training_files = len(fnmatch.filter(os.listdir(args.input_dir), f"train_*.{args.data_format}"))
    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")
        print(f"Found {args.n_training_files} training shards", flush=True)

    args.device_max_steps = args.max_steps

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
        print(args, flush=True)
    return device, local_rank


def prepare_model_and_optimizer(args, device, local_rank, checkpoint, tokenizer):
    config = BertConfig(args.config_file)
    model = T5(config, tokenizer.token_to_id("[PAD]"))

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(config.to_dict())
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)


    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)

    params = list(model.named_parameters())
    no_decay = ['bias', 'layer_norm', '_embedding']
    decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay)]
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
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    scheduler = cosine_schedule_with_warmup(optimizer, int(args.device_max_steps * args.warmup_proportion), args.device_max_steps, 0.1)

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    return model, config, optimizer, scheduler, grad_scaler


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


def training_epoch(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        grad_scaler,
        global_step,
        epoch,
        args,
        device,
        max_local_steps,
        valid_dataloader,
    ):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    if is_main_process():
        train_iter = tqdm(train_dataloader, desc="Train iteration", initial=global_step, total=args.device_max_steps)
    else:
        train_iter = train_dataloader

    for local_step, batch in enumerate(train_iter):
        input_ids, attention_mask, target_ids = [t.to(device, non_blocking=True) for t in batch]
        input_ids, target_ids = input_ids.t(), target_ids.t()

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, perplexity, accuracy = model(input_ids, target_ids, attention_mask)

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

        return_value = grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if return_value is None:
            continue

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
                    "stats/grad_scale": grad_scaler.get_scale()
                },
                step=global_step,
            )
        if (global_step % args.save_every == 0):
            save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args)
            validation_epoch(model, valid_dataloader, epoch, args, device, global_step)
            model = model.train()


        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps or local_step >= max_local_steps - 1:
            return global_step

    return global_step


@torch.no_grad()
def validation_epoch(model, valid_dataloader, epoch, args, device, global_step):
    model = model.eval()

    if is_main_process():
        valid_iter = tqdm(valid_dataloader, desc="Validation iteration")
    else:
        valid_iter = valid_dataloader

    losses, perplexities, accuracies = [], [], []
    with torch.no_grad():
        for batch in valid_iter:
            input_ids, attention_mask, target_ids = [t.to(device, non_blocking=True) for t in batch]
            input_ids, target_ids = input_ids.t(), target_ids.t()

            with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                loss, perplexity, accuracy = model(input_ids, target_ids, attention_mask)

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
            },
            step=global_step,
        )


def save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    checkpoint_dir = f"{args.output_dir}/step{global_step}"
    checkpoint_path = f"{checkpoint_dir}/model.bin"
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
        if os.path.exists(checkpoint_path):
            os.rename(checkpoint_path, f"{checkpoint_path}_tmp")

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path
        )

    return checkpoint_path


def load_datasets(args, tokenizer, epoch, device):
    train_index = (get_rank() + epoch * get_world_size()) % args.n_training_files
    train_seed = args.seed + get_rank() + epoch * get_world_size()

    train_path = f"{args.input_dir}/train_{train_index:05d}.{args.data_format}"
    train_data = Dataset(train_path, tokenizer, seq_length=args.seq_length, mask_p=args.mask_p, short_p=args.short_p)
    print(f"Loaded training file {train_index} on GPU {get_rank()}", flush=True)

    valid_data = Dataset(
        f"{args.input_dir}/validation.{args.data_format}",
        tokenizer,
        seq_length=args.seq_length,
        mask_p=args.mask_p,
        short_p=args.short_p,
    )
    print(f"Loaded validation file on GPU {get_rank()}", flush=True)

    min_length = torch.tensor(len(train_data) // args.batch_size, dtype=torch.long, device=device)
    print(f"Min length: {min_length}", flush=True)
    torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=3,
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
        collate_fn=CollateFunctor(tokenizer.token_to_id("[PAD]"))
    )

    valid_dataloader = DataLoader(
        valid_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=3,
        generator=torch.Generator().manual_seed(42),
        drop_last=True,
        pin_memory=True,
        collate_fn=CollateFunctor(tokenizer.token_to_id("[PAD]"))
    )

    return train_dataloader, valid_dataloader, min_length


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
    device, local_rank = setup_training(args)
    model, config, optimizer, scheduler, grad_scaler = prepare_model_and_optimizer(args, device, local_rank, checkpoint, tokenizer)
    
    for epoch in count(initial_epoch):
        train_dataloader, valid_dataloader, min_length = load_datasets(args, tokenizer, epoch, device)
        if epoch == 0: # sanity check
            validation_epoch(model, valid_dataloader, epoch, args, device)

        global_step = training_epoch(
            model,
            train_dataloader,
            optimizer, scheduler,
            grad_scaler,
            global_step,
            epoch,
            args,
            device,
            min_length,
            valid_dataloader,
        )
        save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args)

        if (epoch + 1) % args.validate_every == 0 or global_step == args.device_max_steps:
            validation_epoch(model, valid_dataloader, epoch, args, device)

        if global_step >= args.device_max_steps:
            break
