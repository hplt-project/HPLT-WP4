import os
import json
from smart_open import open
import argparse
import re
from collections import Counter

from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors, Regex, normalizers


def parse_args():
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--input_dir', type=str, required=True, default="/scratch/project_465000498/processed_data/nn/shards")
    parser.add_argument('--output_dir', type=str, required=True, default="/scratch/project_465000498/processed_data/nn")
    parser.add_argument('--num_sampled_files', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=2**15, help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=10, help='Minimal number of occurences of every candidate subword')
    parser.add_argument('--do_calculate_stats', action='store_true', help='Calculate statistics about the dataset')
    args = parser.parse_args()

    return args


def initialize_tokenizer(args):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    special_tokens += [f"[MASK_{i}]" for i in range(1, 100)]

    number_of_training_shards = len([filename for filename in os.listdir(args.input_dir) if filename.endswith(".jsonl.gz") and "train" in filename])
    if number_of_training_shards == 1:
        args.vocab_size //= 2

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([

        # split on any whitespace character
        pre_tokenizers.Metaspace(),

        # split digits
        pre_tokenizers.Split(
            Regex("▁{0,1}\d{1}"),
            behavior="isolated",
            invert=False
        ),

        # split on any punctuation character
        pre_tokenizers.Split(
            Regex("▁{0,1}[^\w▁]"),
            behavior="isolated",
            invert=False
        ),

        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(Regex(" *\n"), "\n"),
        normalizers.Replace(Regex("\n{2,}"), "██ "),
        normalizers.Replace(Regex("\n"), "█ "),
        normalizers.Replace(Regex(" +"), " "),
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(add_prefix_space=False, use_regex=False),
        decoders.Metaspace(add_prefix_space=False),
        decoders.Strip(' ', 1, 0),
        decoders.Replace(Regex("█ "), "\n"),
        decoders.Replace(Regex("█"), "\n")
    ])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=['█'] + pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=args.min_frequency,
        continuing_subword_prefix='',
        show_progress=True
    )

    return tokenizer, trainer


def calculate_stats(tokenizer, args):
    import tokenization_scorer

    counter, n_words = Counter(), 0
    all_tokens = []
    for document in open(f"{args.input_dir}/validation.jsonl.gz", "rt"):
        text = json.loads(document)
        text = text.rstrip()
        text = limit_repetitions(text)
        if len(text) > 0:
            n_words += len(text.split())
            tokens = tokenizer.encode(text).tokens
            counter.update(tokens)
            all_tokens += tokens

    sorted_subwords = counter.most_common()

    n_subwords = sum(freq for _, freq in sorted_subwords)
    print(f"Average splits per word: {n_subwords / n_words:.3f}", flush=True)

    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]
    renyi_score = tokenization_scorer.score(' '.join(all_tokens), metric="renyi", power=2.5)

    print(f"F_{{95%}} is {f_95}\n")
    print(f"Renyi score is {renyi_score}\n")

    with open(f"{args.output_dir}/tokenizer_stats.txt", "w") as f:
        f.write(f"Vocabulary size: {args.vocab_size}\n")
        f.write(f"Average splits per word: {n_subwords / n_words:.3f}\n")
        f.write(f"F_{{95%}} is {f_95}\n")
        f.write(f"Renyi score is {renyi_score}\n\n")
        sorted_subwords_str = '\n\t'.join(f"{freq}: {subword}" for subword, freq in sorted_subwords)
        f.write(f"Sorted subwords:\n\t{sorted_subwords_str}\n")


if __name__ == "__main__":
    args = parse_args()

    print(f"Initializing a WordPiece tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer", flush=True)
    def limit_repetitions(s):
        return re.sub(r'(\S)(\1{7,})', lambda m: m.group(1) * 8, s)

    def iterator(dir_path, num_sampled_files):
        for filename in sorted(os.listdir(dir_path)):
            if num_sampled_files <= 0:
                break

            if not filename.endswith(".jsonl.gz") or "train" not in filename:
                continue

            for line in open(os.path.join(dir_path, filename), "rt"):
                text = json.loads(line)
                text = text.rstrip()
                text = limit_repetitions(text)
                if len(text) == 0:
                    continue
                yield text

            num_sampled_files -= 1

    tokenizer.train_from_iterator(iterator(args.input_dir, args.num_sampled_files), trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(f"{args.output_dir}/tokenizer.json")

    s = """John Bonham (1948–1980) var ein britisk musikar og låtskrivar, mest kjend som trommeslagar i Led Zeppelin.\nBonham er velrenommert for snøggleiken, krafta, den kjappe høgrefoten, den særeigne stilen og kjensla si for grooven. Han vert rekna som ein av dei største trommeslagarane i rockehistoria av mange trommeslagarar, andre musikarar og musikkekspertar."""
    e = tokenizer.encode(s)
    print(tokenizer.decode(e.ids))
    for i in e.ids:
        print(i, tokenizer.id_to_token(i))

    if args.do_calculate_stats:
        calculate_stats(tokenizer, args)

