#!/bin/bash
#SBATCH --job-name=upload
#SBATCH --account=nn10029k
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1750
#SBATCH --partition=normal
SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"
for lang in bel_Cyrl ukr_Cyrl cmn_Hans vie_Latn por_Latn cat_Latn tha_Thai tam_Taml ind_Latn pol_Latn spa_Latn jpn_Jpan ita_Latn nld_Latn
    do
        echo ${lang}
        srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 upload_branch.py --path /cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models --lang ${lang}
    done