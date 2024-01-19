# this script is used to shard the data into smaller files
# it takes in the input directory, output directory, and the shard size
# the input directory is a directory containing N the jsonl.zst files
# the shard size is the approximate size of each shard in MB (after compression)
# the output directory is the directory where the K sharded files for training and an extra validation file will be stored
# K is calculated to be a power of 2

import argparse
import os
import gzip
import math
import subprocess
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, default="/scratch/project_465000498/one/cleaned/nn")
    parser.add_argument('--output_dir', type=str, required=True, default="/scratch/project_465000498/processed_data/nn")
    parser.add_argument('--shard_size_mb', type=int, required=True, default=512)
    return parser.parse_args()


def count_total_size(input_dir):
    total_size = 0
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl.zst"):
            continue

        total_size += os.path.getsize(os.path.join(input_dir, filename))

    total_size /= 1024 * 1024
    return total_size


def schedule(input_dir, output_dir, shard_size):
    total_size = count_total_size(input_dir)
    number_of_shards = 2 ** math.floor(math.log(total_size / (shard_size / 2), 2))
    actual_shard_size = total_size / number_of_shards

    print(f"Total size: {total_size:.2f} MB")
    print(f"Number of shards: {number_of_shards} files, each of roughly {actual_shard_size*2:.2f} MB", flush=True)
    
    # recursively remove the output directory
    shutil.rmtree(output_dir, ignore_errors=True)

    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    shard_dir = os.path.join(output_dir, "text_shards")
    os.makedirs(shard_dir, exist_ok=True)

    # schedule shard workers
    num_scheduled_shards = 0.0
    current_input_files, current_input_file_size = [], 0
    has_scheduled_validation = False
    shard_job_ids = []

    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith(".jsonl.zst")]
    for i, filename in enumerate(sorted(filenames)):
        current_input_files.append(filename)
        current_input_file_size += os.path.getsize(os.path.join(input_dir, filename)) / 1024 / 1024

        if current_input_file_size < 32 * actual_shard_size and i != len(filenames) - 1:
            continue

        num_shards = current_input_file_size / actual_shard_size
        shards = list(range(int(num_scheduled_shards), int(num_scheduled_shards + num_shards)))
        num_scheduled_shards += num_shards

        print(f"Scheduling [{', '.join(current_input_files)}] to shards [{', '.join(map(str, shards))}]", flush=True)

        # schedule shards with sbatch
        current_input_files = [os.path.join(input_dir, filename) for filename in current_input_files]
        command = f"sbatch shard_worker.sh {','.join(current_input_files)} {shard_dir} {','.join(map(str, shards))} {'--create_validation' if not has_scheduled_validation else ''}"
        bash_output = subprocess.check_output(command, shell=True)
        print(bash_output.decode("utf-8"))
        has_scheduled_validation = True

        job_id = bash_output.decode("utf-8").split()[-1]
        shard_job_ids.append(job_id)

        current_input_files, current_input_file_size = [], 0


    # schedule tokenizer training
    print(f"Scheduling tokenizer training", flush=True)

    command = f"sbatch --dependency=afterok:{':'.join(shard_job_ids)} train_tokenizer.sh {shard_dir} {output_dir}"
    bash_output = subprocess.check_output(command, shell=True)
    print(bash_output.decode("utf-8"))
    tokenizer_job_id = bash_output.decode("utf-8").split()[-1]

    # schedule shard tokenization, batch together 64 jobs
    tokenized_shard_dir = os.path.join(output_dir, "tokenized_shards")
    os.makedirs(tokenized_shard_dir, exist_ok=True)

    for shard_batch in range(math.ceil(number_of_shards / 64)):
        print(f"Scheduling tokenization of shards, batch {shard_batch}, shards {shard_batch * 64} to {min(number_of_shards - 1, (shard_batch + 1) * 64 - 1)}", flush=True)

        input_shard_files = []
        output_shard_files = []
        for shard in range(64):
            if shard_batch * 64 + shard >= number_of_shards:
                break

            input_shard_files.append(os.path.join(shard_dir, f"train_{shard_batch * 64 + shard:05d}.jsonl.gz"))
            output_shard_files.append(os.path.join(tokenized_shard_dir, f"train_{shard_batch * 64 + shard:05d}.pt.gz"))

        input_shard_files = ",".join(input_shard_files)
        output_shard_files = ",".join(output_shard_files)
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        command = f"sbatch --dependency=afterok:{tokenizer_job_id} tokenize_shards.sh {input_shard_files} {output_shard_files} {tokenizer_path}"
        os.system(command)

    # schedule validation tokenization
    print(f"Scheduling tokenization of the validation set", flush=True)

    input_shard_file = os.path.join(shard_dir, "validation.jsonl.gz")
    output_shard_file = os.path.join(tokenized_shard_dir, "validation.pt.gz")
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    command = f"sbatch --dependency=afterok:{tokenizer_job_id} tokenize_shards.sh {input_shard_file} {output_shard_file} {tokenizer_path}"
    os.system(command)

if __name__ == "__main__":
    args = parse_args()
    schedule(args.input_dir, args.output_dir, args.shard_size_mb)
