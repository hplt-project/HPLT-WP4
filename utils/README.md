## Utils

Utils to validate data

`rename.py` # rename input folders to make it compatible with v1 (no "-" and "_" in language names, short enough to be seen in `squeue` by default)
`reorder.sh` # check that training and validation (tokenized) files are readable, remove broken training files, rename in the correct order from 0 to n

Utils to upload models to Huggingface

`mass_copy.py` # ensure correct naming, copy model card, replace English with a correct language in it
`upload.py` # upload a single model 
