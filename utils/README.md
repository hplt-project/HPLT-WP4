## Utils

### Utils to validate data

`count_num_docs.sh` # draft of a script to count number of documents in any of our files

`fix_validation.sh` # if not enough files in validation, move them from a train file

`rename.py` # not used for 3.0; rename input folders to make it compatible with v1 (no "-" and "_" in language names, short enough to be seen in `squeue` by default)

### Utils to upload models to Huggingface

#### Initial upload

`mass_upload.sh` # upload all models in a folder

`collection.py` # add to a HF collection

`upload_branch.py` # upload all branches

#### Fix if something went wrong / if you want something custom

`delete_branch.py` # delete one or many branches

`delete_file.py` # delete a single file from all models in a HF collection

`edit_readme.py` # fix README in a `huggingface_prototype`

`fix_missing_checkpoints.py` # fix something in all branches based on the last commit message

`rename_repo.py` # rename HF repository

`upload.py` # upload a single model 

`update_modeling.sh` # update custom code in all checkpoints

`update_readme.py` # fix READMEs in all models in a HF collection

`update_single_files.py` # update a single file for all branches on HF

`upload_branch.py` # update branches for a single language

`upload_file.py` # update a single file for the main branch on HF

### Plots

`win_rates.ipynb` # reproduce plot from the ACL'25 paper

