#!/bin/bash

#SBATCH --job-name=SCHEDULER
#SBATCH --account=nn10029k
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --output=/cluster/work/projects/nn9851k/mariiaf/hplt/logs/schedule-%j.out


if [ -z $SLURM_JOB_ID ]; then
    mkdir -p /cluster/work/projects/nn9851k/mariiaf/hplt/logs
    sbatch --job-name "${1}-SCHEDULE" --output "/cluster/work/projects/nn9851k/mariiaf/hplt/logs/${1}-schedule-%j.out" "$0" "$@"
    exit
fi

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

LANGS=${@}
# run the script
for lang in $LANGS
do
  echo "Running schedule.py"
  python3 start_from_tokenizing.py --language $lang --output_dir /cluster/work/projects/nn9851k/mariiaf/hplt/ --tokenize_train --olivia
done