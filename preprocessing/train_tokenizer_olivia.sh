#!/bin/bash

#SBATCH --account=nn10029k
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small

# ^ memory > 2G is really needed when saving the tokenizer
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"

# process arguments
## input and output directories
INPUT_DIR=${1}
OUTPUT_DIR=${2}
OTHER_ARGS=${@:3}

# run the script
echo "Running train_tokenizer.py input_dir ${INPUT_DIR} output_dir ${OUTPUT_DIR} ${OTHER_ARGS}"
srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 train_tokenizer.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} ${OTHER_ARGS}
