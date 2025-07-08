#! /bin/bash

# Mass uploading intermediate checkpoints to HF as branches
#
CHECKPOINTS_PATH=${1}
for swhLatn in swhLatn_3125 swhLatn_6250 swhLatn_9375 swhLatn_12500 swhLatn_15625 swhLatn_18750 swhLatn_21875 swhLatn_25000 swhLatn_28125       	
	do
	python3 upload_branch.py ${CHECKPOINTS_PATH} ${swhLatn}
	done

