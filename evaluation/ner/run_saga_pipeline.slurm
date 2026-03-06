#!/bin/bash
#SBATCH --job-name=schedule
#SBATCH --account=nn9851k
#SBATCH --partition=normal
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=1
#SBATCH --qos=devel

module --quiet purge  # Reset the modules to the system default
module --force swap StdEnv Zen2Env
module use -a /cluster/shared/nlpl/software/eb/etc/all/


module load nlpl-transformers/4.38.2-foss-2022b-Python-3.10.8

module list    # For easier debugging

python3 ner_pipeline_3_0.py