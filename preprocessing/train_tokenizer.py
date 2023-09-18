import os
import json
from smart_open import open
import argparse
from collections import Counter

from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors


def parse_args():
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--input_dir', type=str, required=True, help='Specify the input filename')
    parser.add_argument('--validation_file', type=str, required=True, help='Specify the input filename')
    parser.add_argument('--tokenizer_path', type=str,required=True, help='Specify the output filename')
    parser.add_argument('--num_sampled_files', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=2**15, help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=10, help='Minimal number of occurences of every candidate subword')
    parser.add_argument('--do_calculate_stats', action='store_true', help='Calculate statistics about the dataset')
    args = parser.parse_args()

    return args


def initialize_tokenizer(args):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    special_tokens += [f"[MASK_{i}]" for i in range(1, 100)]

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=True),
        pre_tokenizers.Digits(individual_digits=True)
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=args.min_frequency,
        continuing_subword_prefix='',
        show_progress=True
    )

    return tokenizer, trainer


def calculate_stats(tokenizer, args):
    import tokenization_scorer

    counter, n_words = Counter(), 0
    all_tokens = []
    for document in open(args.validation_file, mode='rt'):
        text = json.loads(document)["text"]
        if len(text) > 0:
            n_words += len(text.split())
            tokens = tokenizer.encode(text).tokens
            counter.update(tokens)
            all_tokens += tokens

    sorted_subwords = counter.most_common()

    n_subwords = sum(freq for _, freq in sorted_subwords)
    print(f"Average splits per word: {n_subwords / n_words:.3f}", flush=True)
    print(f"Number of subwords: {n_subwords}", flush=True)

    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]
    renyi_score = tokenization_scorer.score(' '.join(all_tokens), metric="renyi", power=2.5)

    print(f"F_{{95%}} is {f_95}\n")
    print(f"Renyi score is {renyi_score}\n")


if __name__ == "__main__":
    args = parse_args()

    print(f"Initializing a WordPiece tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer", flush=True)
    def iterator(dir_path, num_sampled_files):
        for filename in os.listdir(dir_path):
            if num_sampled_files <= 0:
                break

            if not filename.endswith(".jsonl") and not filename.endswith(".jsonl.gz"):
                continue

            for line in open(os.path.join(dir_path, filename), "rt"):
                document = json.loads(line)
                text = document["text"]
                text = text.rstrip()
                if len(text) == 0:
                    continue
                yield text

            num_sampled_files -= 1

    tokenizer.train_from_iterator(iterator(args.input_dir, args.num_sampled_files), trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.tokenizer_path)

    if args.do_calculate_stats:
        calculate_stats(tokenizer, args)
