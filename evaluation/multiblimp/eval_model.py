from glob import glob
import argparse
import sys
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from load_model import load_hf_model
from norsk import score_norsk
from score import score_tse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="LM to evaluate")
    parser.add_argument(
        "--data_filename", help="Minimal pair file", default='eng/data.tsv',
    )
    parser.add_argument("--src_dir", help="Source directory", default="./")
    parser.add_argument("--results_dir", help="Dir to write results to", default=None)
    parser.add_argument(
        "--cache_dir", help="(optional) HF cache dir", default="/scratch/project_465002310/cache/"
    )
    parser.add_argument(
        "--hf_token",
        help="Huggingface token (file or token itself)",
        default=None,
    )
    parser.add_argument('--mask_1', default='')
    parser.add_argument('--mask_2', default='')
    parser.add_argument('--eos', default='[SEP]')
    args = parser.parse_args()
    print(args, flush=True)
    if not args.data_filename[:3] == 'nob':
        data_dir = os.path.dirname(
            hf_hub_download(repo_id="jumelet/multiblimp", filename=args.data_filename, repo_type="dataset"),
        )
        args.data_dir = data_dir
        print(args.data_dir, flush=True)
    else:
        data = load_dataset('ltg/ask-gec', split='test')
        data = data.filter(lambda example: example["source"] != example["correction"])
    sys.path.append(args.src_dir)


    if args.hf_token is not None:
        with open(args.hf_token) as f:
            hf_token = f.read().strip()
    else:
        hf_token = args.hf_token
    is_encoder = bool(args.mask_1)
    lm = load_hf_model(
        args.model_name,
        no_cache=False,
        is_encoder=is_encoder,
        token=hf_token, 
        cache_dir=os.path.expanduser(args.cache_dir),
        device_map=0,
    )
    print("Model loaded")
    if not args.data_filename[:3] == 'nob':
        pair_files = glob(os.path.join(args.data_dir, "*.tsv"))
    else:
        pair_files = [data]
    for fn in sorted(pair_files):
        if isinstance(fn, str):
            df = score_tse(lm, fn=fn, is_encoder=is_encoder, mask_1=args.mask_1, mask_2=args.mask_2)
            results_dir = (
                os.path.join("model_results", os.path.split(args.model_name)[-1])
                if args.results_dir is None
                else args.results_dir
            )
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)

            score_fn = os.path.join(
                results_dir, f'{os.path.split(args.model_name)[-1]}-{os.path.split(args.data_filename)[0]}.tsv',
            )
            df.to_csv(score_fn, sep="\t")
            print(f"acc {round(df[df.delta>0].shape[0]/df.shape[0],3)}", flush=True)
        else:
            args.mask_1 = ' ' + args.mask_1 + ' '
            score_norsk(fn, lm, mask_symbol=args.mask_1, eos=args.eos)
