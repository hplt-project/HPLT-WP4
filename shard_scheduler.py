from glob import glob
import math
import os
import shutil
import subprocess

class ShardScheduler:
    def __init__(self, args):
        # schedule shard workers
        self.max_n_jobs = args.max_n_jobs
        self.input_dir = args.input_dir
        self.wds_best = args.wds_best
        if self.wds_best:
            print("Using documents scored 10 only", flush=True)
            self.filenames = [os.path.join(args.input_dir, filename) for filename in os.listdir(args.input_dir) if
                        filename.endswith(".jsonl.zst") and (filename.startswith("10_") or filename.startswith("9_"))]
        else:
            self.filenames = glob(os.path.join(self.input_dir, "*.jsonl.zst"))
            print(self.input_dir)
            print(os.listdir(self.input_dir))
            print(self.filenames)
        if not args.n_training_documents:
            total_size = self._count_total_size()
            self.number_of_shards = 2 ** max(
                0, math.floor(
                math.log(total_size / (args.shard_size_mb / 2), 2),
            ),
            )
            self.actual_shard_size = total_size / self.number_of_shards
            print(f"Total size: {total_size:.2f} MB", flush=True)
        else:
            n_input_files = len(self.filenames)
            self.docs_to_pick = args.n_training_documents // n_input_files
            print(f"Number of documents to pick from each file: {self.docs_to_pick}", flush=True)
            self.number_of_shards = min(n_input_files, self.max_n_jobs)
            print(f'number of shards: {self.number_of_shards}')
        self.shard_dir = os.path.join(args.output_dir, "text_shards")
        self.shard_job_ids = []
        self.num_scheduled_shards = 0.0
        self.language = args.language
        self.current_input_files = []
        self.has_scheduled_validation = False
        self.output_dir = args.output_dir
        self.logs_dir = os.path.join(self.output_dir, "logs/")
        self.sample_power = args.sample_power
        self.n_training_documents = args.n_training_documents
        self.filenames.sort(key=os.path.getsize, reverse=True) # ensure enough validation docs in the 1st
        self.filenames_not_first = self.filenames[1:]
        self.filenames_not_first.reverse()
        self.filenames = [self.filenames[0]] + self.filenames_not_first # escape the situation of too large first batch
        self.olivia = args.olivia
        

    def _count_total_size(self):
        total_size = 0
        for filename in os.listdir(self.input_dir):
            if self.wds_best and not (filename.startswith("10_") or filename.startswith("9_")):
                continue
            elif not self.wds_best:
                if not filename.endswith(".jsonl.zst"):
                    continue

            total_size += os.path.getsize(os.path.join(self.input_dir, filename)) # in bytes 

        total_size /= 1024 * 1024 # in MiB
        return total_size


    def _schedule_shards(self, shards):
        # schedule shards with sbatch
        print(
            f"Scheduling [{', '.join(self.current_input_files)}] to shards [{', '.join(map(str, shards))}]",
            flush=True,
        )
        script = "shard_worker"
        if self.olivia:
            script = "shard_worker_olivia"
        command = f"sbatch --job-name {self.language}-SHARD --chdir preprocessing --output {self.logs_dir}{self.language}-shard-%j.out preprocessing/{script}.sh" +  \
        f" {','.join(self.current_input_files)} {self.shard_dir} {','.join(map(str, shards))} {self.sample_power}"
        if not self.has_scheduled_validation:
            command += ' --create_validation'
        if self.n_training_documents:
            command += f" --docs_to_pick {self.docs_to_pick}"
        bash_output = subprocess.check_output(command, shell=True)
        print(bash_output.decode("utf-8"))
        self.has_scheduled_validation = True
        job_id = bash_output.decode("utf-8").split()[-1]
        self.shard_job_ids.append(job_id)
        self.current_input_files = []


    def _shard_few_zst_files(self):
        current_input_file_size = 0
        for i, filename in enumerate(self.filenames):
            self.current_input_files.append(filename)
            if not self.n_training_documents:
                current_input_file_size += os.path.getsize(filename) / 1024 / 1024

                if current_input_file_size < 32 * self.actual_shard_size and i != len(
                        self.filenames) - 1:
                    continue

                num_shards = current_input_file_size / self.actual_shard_size 
            else:
                num_shards = 1
            shards = list(range(int(self.num_scheduled_shards),
                                int(self.num_scheduled_shards + num_shards)))
            self.num_scheduled_shards += num_shards
            # if we are at the last file, make sure all shards are created
            if i == len(self.filenames) - 1 and shards[-1] < self.number_of_shards - 1:
                shards += list(range(shards[-1], self.number_of_shards))
            self._schedule_shards(shards)
            current_input_file_size = 0

    def _shard_many_zst_files(self):
        n_input_files = len(self.filenames)
        files_in_shards = n_input_files // self.number_of_shards
        shard_size = files_in_shards + (n_input_files % self.number_of_shards)
        for i, filename in enumerate(self.filenames):
            if len(self.current_input_files) < shard_size:
                self.current_input_files.append(filename)
            else:
                shards = list(range(int(self.num_scheduled_shards),
                                int(self.num_scheduled_shards + 1)))
                self.num_scheduled_shards += 1
                # if we are at the last file, make sure all shards are created
                if i == len(self.filenames) - 1 and shards[-1] < self.number_of_shards - 1:
                    shards += list(range(shards[-1], self.number_of_shards))
                self._schedule_shards(shards)
                shard_size = files_in_shards

    def schedule(self):
        # recursively remove the output directory
        shutil.rmtree(self.output_dir, ignore_errors=True)
        # make sure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.shard_dir, exist_ok=True)
        if len(self.filenames) <= self.max_n_jobs:
            self._shard_few_zst_files()
            assert self.number_of_shards == self.num_scheduled_shards
        else:
            self._shard_many_zst_files()
        print(
            f"Number of shards: {self.num_scheduled_shards} files",
            flush=True,
        )