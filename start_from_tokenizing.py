# this script is used to schedule the preprocessing and training jobs for a language
# it takes as input the language, the output from sharding and training tokenizer directory
# it then schedules the shard tokenization and the BERT training

import argparse
import os
import math
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True, default="itaL")
    parser.add_argument('--output_dir', type=str, default="/scratch/project_465002259/hplt-3-0-t5/")
    parser.add_argument('--tokenize_train', action='store_true')
    parser.add_argument('--run_training', action='store_true')
    return parser.parse_args()


def schedule(language, output_dir):
    output_dir = os.path.join(output_dir, language)
    shard_dir = os.path.join(output_dir, "text_shards")
    # schedule shard tokenization, batch together ntasks jobs
    tokenized_shard_dir = os.path.join(output_dir, "tokenized_shards")
    os.makedirs(tokenized_shard_dir, exist_ok=True)
    tokenization_job_ids = [] # MaxSubmit on standard is 120, see sacctmgr list user $USER withassocq
    if args.tokenize_train:
        shard_files = os.listdir(shard_dir)
        number_of_shards = len(shard_files)
        print(f"Number of shards: {number_of_shards}")
        if "validation.jsonl.gz" in shard_files:
            number_of_shards = number_of_shards - 1
        ntasks = 64
        for shard_batch in range(math.ceil(number_of_shards / ntasks)):
            print(f"Scheduling tokenization of shards, batch {shard_batch}, shards {shard_batch * ntasks} to {min(number_of_shards - 1, (shard_batch + 1) * ntasks - 1)}", flush=True)

            input_shard_files = []
            output_shard_files = []
            for shard in range(ntasks):
                if shard_batch * ntasks + shard >= number_of_shards:
                    break
                
                output_shard_file = os.path.join(tokenized_shard_dir, f"train_{shard_batch * ntasks + shard:05d}.pt.gz")
                if not os.path.exists(output_shard_file):
                    input_shard_files.append(os.path.join(shard_dir, f"train_{shard_batch * ntasks + shard:05d}.jsonl.gz"))
                    output_shard_files.append(output_shard_file)
            n_files_in_batch = len(input_shard_files)
            input_shard_files = ",".join(input_shard_files)
            output_shard_files = ",".join(output_shard_files)
            tokenizer_path = os.path.join(output_dir, "tokenizer.json")
            command = f"sbatch --job-name {language}-TOKENIZE --ntasks-per-node={n_files_in_batch} --chdir preprocessing --output /scratch/project_465002259/hplt-3-0-t5/logs/{language}-tokenize-%j.out preprocessing/tokenize_shards.sh {input_shard_files} {output_shard_files} {tokenizer_path}"
            bash_output = subprocess.check_output(command, shell=True).decode("utf-8")
            print(bash_output)
            tokenization_job_ids.append(bash_output.split()[-1])

    # schedule validation tokenization
    print(f"Scheduling tokenization of the validation set", flush=True)

    input_shard_file = os.path.join(shard_dir, "validation.jsonl.gz")
    output_shard_file = os.path.join(tokenized_shard_dir, "validation.pt.gz")
    if not os.path.exists(output_shard_file):
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        command = f"sbatch --job-name {language}-TOKENIZE --ntasks-per-node=1 --chdir preprocessing --output /scratch/project_465002259/hplt-3-0-t5/logs/{language}-tokenize-%j.out preprocessing/tokenize_shards.sh {input_shard_file} {output_shard_file} {tokenizer_path}"
        bash_output = subprocess.check_output(command, shell=True).decode("utf-8")
        print(bash_output, flush=True)
        tokenization_job_ids.append(bash_output.split()[-1])

    if args.run_training:
        # schedule BERT training
        print(f"Scheduling BERT training", flush=True)
        command = f"sbatch --job-name {language}-BERT --chdir encoder-only --output /scratch/project_465002259/hplt-3-0-t5/logs/{language}-bert-%j.out --dependency=afterok:{':'.join(tokenization_job_ids)} encoder-only/train.sh {language} {output_dir}"
        bash_output = subprocess.check_output(command, shell=True)
        print(bash_output.decode("utf-8"), flush=True)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    schedule(args.language, args.output_dir)
