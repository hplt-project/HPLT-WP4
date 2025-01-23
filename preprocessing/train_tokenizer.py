import time
import os
import json
from smart_open import open
import argparse
import re
from collections import Counter
from gzip import BadGzipFile

from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors, Regex, \
    normalizers

from timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--input_dir', type=str, required=True,
                        default="~/processed_data/nn/shards")
    parser.add_argument('--output_dir', type=str, required=True,
                        default="~/processed_data/nn")
    parser.add_argument('--num_sampled_files', type=int, default=16)
    parser.add_argument('--vocab_size', type=int, default=2 ** 15,
                        help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=10,
                        help='Minimal number of occurences of every candidate subword')
    parser.add_argument('--do_calculate_stats', action='store_true',
                        help='Calculate statistics about the dataset')
    parser.add_argument('--do_japanese_pretokenization', action='store_true',
                        help='Use Japanese pre-tokenization')
    parser.add_argument('--do_korean_pretokenization', action='store_true',
                        help='Use Korean pre-tokenization')
    parser.add_argument('--do_thai_pretokenization', action='store_true',
                        help='Use Thai pre-tokenization')
    parser.add_argument('--do_burmese_pretokenization', action='store_true',
                        help='Use Burmese pre-tokenization')
    parser.add_argument('--do_chinese_pretokenization', action='store_true',
                        help='Use Chinese pre-tokenization')
    args = parser.parse_args()

    return args


def initialize_tokenizer(args):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    special_tokens += [f"[MASK_{i}]" for i in range(1, 100)] + ['█']

    number_of_training_shards = len(
        [filename for filename in os.listdir(args.input_dir) if
         filename.endswith(".jsonl.gz") and "train" in filename])
    if number_of_training_shards == 1:
        args.vocab_size //= 2

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([

        # split on any whitespace character
        pre_tokenizers.Metaspace(
            #add_prefix_space=not args.do_japanese_pretokenization
        ),

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
        decoders.Metaspace(
            #add_prefix_space=False
        ),
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
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=args.min_frequency,
        continuing_subword_prefix='',
        show_progress=True
    )

    return tokenizer, trainer


def calculate_stats(tokenizer, args):
    import tokenization_scorer

    counter, n_words = Counter(), 0
    all_tokens = []
    for i, document in enumerate(
            open(f"{args.input_dir}/validation.jsonl.gz", "rt")):
        text = json.loads(document)
        text = text.rstrip()
        text = limit_repetitions(text)
        if len(text) > 0:
            n_words += len(text.split())
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            counter.update(tokens)
            all_tokens += tokens

            if i == 0:
                print("Example of tokenization:")
                print(text)
                print(tokenizer.decode(encoding.ids))
                for j in encoding.ids:
                    print(j, tokenizer.id_to_token(j))

    sorted_subwords = counter.most_common()

    n_subwords = sum(freq for _, freq in sorted_subwords)
    print(f"Average splits per word: {n_subwords / n_words:.3f}", flush=True)

    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]
    renyi_score = tokenization_scorer.score(' '.join(all_tokens),
                                            metric="renyi", power=2.5)

    print(f"F_{{95%}} is {f_95}\n")
    print(f"Renyi score is {renyi_score}\n")

    with open(f"{args.output_dir}/tokenizer_stats.txt", "w") as f:
        f.write(f"Vocabulary size: {args.vocab_size}\n")
        f.write(f"Average splits per word: {n_subwords / n_words:.3f}\n")
        f.write(f"F_{{95%}} is {f_95}\n")
        f.write(f"Renyi score is {renyi_score}\n\n")
        sorted_subwords_str = '\n\t'.join(
            f"{freq}: {subword}" for subword, freq in sorted_subwords)
        f.write(f"Sorted subwords:\n\t{sorted_subwords_str}\n")


if __name__ == "__main__":
    print(f"Starting training tokenizer at {time.strftime('%Y-%m-%d %H:%M')}. End time: {os.getenv('SLURM_JOB_END_TIME')}")
    print(f"The job {os.getenv('SLURM_JOB_NAME')} numbered {os.getenv('SLURM_JOB_ID')} is running on the nodes {os.getenv('SLURM_JOB_NODELIST')}")
    # start a timer for 71 hours; if the timer runs out, the job will stop and the tokenized files will be saved
    timer = Timer(8 * 60 * 60)
    args = parse_args()

    print(
        f"Initializing a WordPiece tokenizer {time.strftime("%Y-%m-%d %H:%M")}",
        flush=True,
    )
    tokenizer, trainer = initialize_tokenizer(args)

    print(
        f"Training the tokenizer {time.strftime('%Y-%m-%d %H:%M')}",
        flush=True,
    )


    def limit_repetitions(s):
        return re.sub(r'(\S)(\1{7,})', lambda m: m.group(1) * 8, s)


    def iterator(dir_path, num_sampled_files: int, args):
        if args.do_japanese_pretokenization:
            from fugashi import Tagger
            print("Japanese pretokenization")
            tagger = Tagger('-Owakati')
            pretokenize = lambda text: [word.surface for word in tagger(text)]
        elif args.do_korean_pretokenization:
            from kiwipiepy import Kiwi
            print("Korean pretokenization")
            tagger = Kiwi()
            pretokenize = lambda text: [word.form for word in
                                        tagger.tokenize(text)]
        elif args.do_thai_pretokenization:
            from pythainlp.tokenize import word_tokenize
            pretokenize = lambda text: word_tokenize(text, engine="nlpo3",
                                                     keep_whitespace=False)
        elif args.do_burmese_pretokenization:
            import pyidaungsu as pds
            pretokenize = lambda text: pds.tokenize(text, form="word")
        elif args.do_chinese_pretokenization:
            from jieba import cut
            print("Chinese pretokenization")
            pretokenize = lambda text: cut(text, cut_all=False)

        for filename in sorted(os.listdir(dir_path)):
            print(f"Files left: {num_sampled_files} at {time.strftime('%Y-%m-%d %H:%M')}")
            if (num_sampled_files <= 0) or (not timer.has_time_remaining()):
                break

            if not filename.endswith(".jsonl.gz") or "train" not in filename:
                continue

            training_filename = os.path.join(dir_path, filename)
            try:
                for line in open(training_filename, "rt"):
                    text = json.loads(line)
                    text = text.rstrip()
                    text = limit_repetitions(text)
                    if len(text) == 0:
                        continue

                    if not args.do_japanese_pretokenization and not args.do_korean_pretokenization and not args.do_thai_pretokenization and not args.do_burmese_pretokenization and not args.do_chinese_pretokenization:
                        yield text
                    else:
                        for i, paragraph in enumerate(text.split('\n')):
                            for j, segment in enumerate(paragraph.split(' ')):
                                if i > 0 and j == 0:
                                    prefix = '\n'
                                elif j > 0:
                                    prefix = ' '
                                else:
                                    prefix = ''

                                try:
                                    for word in pretokenize(segment):
                                        yield f"{prefix}{word}"
                                except:
                                    continue
            except BadGzipFile as e:
                print(f"{e} on {training_filename}, skipping")
                continue
            num_sampled_files -= 1


    tokenizer.train_from_iterator(
        iterator(args.input_dir, args.num_sampled_files, args),
        trainer
    )

    print("Saving the tokenizer", flush=True)
    tokenizer.save(f"{args.output_dir}/tokenizer.json")

    if args.do_calculate_stats:
        calculate_stats(tokenizer, args)
