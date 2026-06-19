#!/bin/bash

#SBATCH --job-name=SCHEDULER
#SBATCH --account=nn10029k
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH --output=/cluster/work/projects/nn9851k/mariiaf/hplt/logs/schedule-%j.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

TOKENIZER_PATH=${1}
LANG=${2}
OTHER_ARGS=${@:3}
# run the script
echo "Running schedule.py"
python3 start_from_tokenizing.py --language $LANG --output_dir /cluster/work/projects/nn9851k/mariiaf/hplt/ --tokenize_train --olivia --tokenizer $TOKENIZER_PATH $OTHER_ARGS
