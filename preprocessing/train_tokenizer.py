import argparse
from collections import Counter
import tokenization_scorer

from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors


def initialize_tokenizer(args):
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PAR]", "[TAB]"]

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


def calculate_stats(tokenizer, f):
    counter, n_words = Counter(), 0
    tokens = []
    for sentence in f.readlines():
        sentence = sentence.strip()
        n_words += len(sentence.split())
        if len(sentence) > 0:
            tokens = tokenizer.encode(sentence).tokens
            counter.update(tokens)
            tokens += tokens

    sorted_subwords = counter.most_common()

    n_subwords = sum(freq for _, freq in sorted_subwords)
    print(f"Average splits per word: {n_subwords / n_words:.3f}", flush=True)
    print(f"Number of subwords: {n_subwords}", flush=True)

    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]
    renyi_score = tokenization_scorer.score(' '.join(tokens), metric="renyi", power=2.5)

    print(f"F_{{95%}} is {f_95}\n")
    print(f"Renyi score is {renyi_score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--input_path', type=str, default="data/pretrain/bnc/train.md", help='Specify the input filename')
    parser.add_argument('--vocab_path', type=str, default="data/pretrain/bpe.json", help='Specify the output filename')
    parser.add_argument('--vocab_size', type=int, default=2**14, help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=10, help='Minimal number of occurences of every candidate subword')
    args = parser.parse_args()

    print(f"Initializing a WordPiece tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer", flush=True)
    def iterator(file_path: str):
        for line in open(file_path):
            line = line.strip()
            if len(line) == 0:
                continue
            yield line
    tokenizer.train_from_iterator(iterator(args.input_path), trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.vocab_path)

    with open("../data/processed_10M/all.txt") as f:
        calculate_stats(tokenizer, f)

