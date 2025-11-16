# this script is used to schedule the preprocessing and training jobs for a language
# it takes as input the language, the input directory, the output directory, the shard size and optinally the sample power
# it then schedules the shard workers, the tokenizer training, the shard tokenization and the BERT training

import argparse
import os
import math
import subprocess

from shard_scheduler import ShardScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True, default="nn")
    parser.add_argument('--input_dir', type=str, required=True,
                        default="~/one/cleaned/nn")
    parser.add_argument('--output_dir', type=str, required=True,
                        default="~/processed_data/nn")
    parser.add_argument(
        '--shard_size_mb',
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument('--sample_power', type=float, required=False,
                        default=0.0)
    parser.add_argument('--do_calculate_train_tok_stats', action='store_true')
    parser.add_argument(
        '--skip_sharding',
        action='store_true',
        help="start from training a tokenizer",
    )
    parser.add_argument('--wds_best', action='store_true')
    parser.add_argument('--n_training_documents', type=int, default=0)
    parser.add_argument('--run_training', action='store_true')
    parser.add_argument('--run_sharding_only', action='store_true')
    parser.add_argument('--max_n_jobs', default=79)
    return parser.parse_args()


def schedule(args):
    language = args.language
    output_dir = args.output_dir
    shard_scheduler = ShardScheduler(args)
    if not args.skip_sharding:
        shard_scheduler.schedule()
    if not args.run_sharding_only:
        # schedule tokenizer training
        if not os.path.exists(os.path.join(output_dir, 'tokenizer.json')):
            print(f"Scheduling tokenizer training", flush=True)

            additional_args = ""
            if (args.language == "ja") or (args.language.startswith(
                    'jpn')):  # not to be confused with Javanese in 2.0
                additional_args = "--do_japanese_pretokenization"
            elif (args.language == "ko") or (args.language.startswith(
                    'kor')):  # not to be confused with Kongo in 2.0
                additional_args = "--do_korean_pretokenization"
            elif args.language == "my":
                additional_args = "--do_burmese_pretokenization"
            elif args.language.startswith("th"): # has spaces in 3.0 but probably not always
                additional_args = "--do_thai_pretokenization"
            elif args.language.startswith("zh"):
                additional_args = "--do_chinese_pretokenization"
            #elif args.language.startswith("tam"): # actually has spaces
            #    additional_args = "--do_tamil_pretokenization"
            if args.do_calculate_train_tok_stats:
                additional_args += " --do_calculate_stats"

            dependency = ''
            if shard_scheduler.shard_job_ids:
                dependency = f"--dependency=afterok:{':'.join(shard_scheduler.shard_job_ids)}"

            command = f"sbatch --job-name {language}-TRAIN-TOKENIZER --chdir preprocessing --output /scratch/project_465002259/hplt-2-0-output/logs/{language}-train-tokenizer-%j.out {dependency} preprocessing/train_tokenizer.sh {shard_scheduler.shard_dir} {output_dir} {additional_args}"
            bash_output = subprocess.check_output(command, shell=True)
            print(bash_output.decode("utf-8"))
            tokenizer_job_id = bash_output.decode("utf-8").split()[-1]

        # schedule shard tokenization, batch together ntasks jobs
        tokenized_shard_dir = os.path.join(output_dir, "tokenized_shards")
        os.makedirs(tokenized_shard_dir, exist_ok=True)
        tokenization_job_ids = []
        ntasks = 64
        for shard_batch in range(math.ceil(shard_scheduler.number_of_shards / ntasks)):
            print(
                f"Scheduling tokenization of shards, batch {shard_batch}, shards {shard_batch * ntasks} to {min(shard_scheduler.number_of_shards - 1, (shard_batch + 1) * ntasks - 1)}",
                flush=True)

            input_shard_files = []
            output_shard_files = []
            for shard in range(ntasks):
                if shard_batch * ntasks + shard >= shard_scheduler.number_of_shards:
                    break

                input_shard_files.append(os.path.join(shard_scheduler.shard_dir,
                                                    f"train_{shard_batch * ntasks + shard:05d}.jsonl.gz"))
                output_shard_files.append(os.path.join(tokenized_shard_dir,
                                                    f"train_{shard_batch * ntasks + shard:05d}.pt.gz"))
            n_files_in_batch = len(input_shard_files)
            input_shard_files = ",".join(input_shard_files)
            output_shard_files = ",".join(output_shard_files)
            tokenizer_path = os.path.join(output_dir, "tokenizer.json")
            command = f"sbatch --job-name {language}-TOKENIZE --ntasks-per-node={n_files_in_batch} --chdir preprocessing --output /scratch/project_465002259/hplt-2-0-output/logs/{language}-tokenize-%j.out --dependency=afterok:{tokenizer_job_id} preprocessing/tokenize_shards.sh {input_shard_files} {output_shard_files} {tokenizer_path}"
            bash_output = subprocess.check_output(command, shell=True).decode(
                "utf-8")
            print(bash_output)
            tokenization_job_ids.append(bash_output.split()[-1])

        # schedule validation tokenization
        print(f"Scheduling tokenization of the validation set", flush=True)

        input_shard_file = os.path.join(shard_scheduler.shard_dir, "validation.jsonl.gz")
        output_shard_file = os.path.join(tokenized_shard_dir, "validation.pt.gz")
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        command = f"sbatch --job-name {language}-TOKENIZE --ntasks-per-node=1 --chdir preprocessing --output /scratch/project_465002259/hplt-2-0-output/logs/{language}-tokenize-%j.out --dependency=afterok:{tokenizer_job_id} preprocessing/tokenize_shards.sh {input_shard_file} {output_shard_file} {tokenizer_path}"
        bash_output = subprocess.check_output(command, shell=True).decode("utf-8")
        print(bash_output)
        tokenization_job_ids.append(bash_output.split()[-1])
        if args.run_training:
            # schedule BERT training
            print(f"Scheduling BERT training", flush=True)
            command = f"sbatch --job-name {language}-BERT --chdir encoder-only --output /scratch/project_465002259/hplt-2-0-output/logs/{language}-bert-%j.out --dependency=afterok:{':'.join(tokenization_job_ids)} encoder-only/train.sh {language} {output_dir}"
            bash_output = subprocess.check_output(command, shell=True)
            print(bash_output.decode("utf-8"), flush=True)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    schedule(args)
