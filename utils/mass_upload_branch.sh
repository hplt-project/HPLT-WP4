#! /bin/bash

# Mass uploading intermediate checkpoints to HF as branches
#
CHECKPOINTS_PATH=${1}
for checkpoint_path in $CHECKPOINTS_PATH/*
do
    for step in step3125 step6250 step9375 step12500 step15625 step18750 step21875 step25000 step28125       	
    	do
		python3 upload_branch.py ${checkpoint_path} ${step}
	done
done
