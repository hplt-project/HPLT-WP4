import os.path
import subprocess

from constants import LANGS_MAPPING

if __name__ == '__main__':
    for langv2, langv1 in LANGS_MAPPING.items():
      print(langv2)
      try:
        bash_output = subprocess.check_output(f"grep {langv2} ner_results.tsv", shell=True)
        print(bash_output.decode("utf-8"))
      except subprocess.CalledProcessError:
        model_path = f"/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/{langv2}/"
        if not os.path.exists(model_path):
          model_path = f"/scratch/project_465001386/hplt-2-0-output/hplt_hf_models/{langv2}_31250/"
        command = f"sbatch --job-name {langv2}-NER --output /scratch/project_465001386/hplt-2-0-output/logs/ner/{langv2}-ner-%j.out run_ner.slurm {model_path} wikiann/{langv1} hplt_{langv2}"
        bash_output = subprocess.check_output(command, shell=True)
        print(bash_output.decode("utf-8"))