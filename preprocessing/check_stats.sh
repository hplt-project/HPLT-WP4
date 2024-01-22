#!/bin/bash

#SBATCH --account=project_465000498
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# Load the virtual environment
source /project/project_465000144/pytorch_1.13.1/bin/activate

# process arguments
## input and output directories
INPUT_FILE=${1}
OUTPUT_FILE=${2}

# run the script
python3 check_stats.py --input_file ${INPUT_FILE} --output_file ${OUTPUT_FILE}
