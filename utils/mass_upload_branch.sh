#! /bin/bash

# Mass uploading intermediate checkpoints to HF as branches
#
for i in hplt*/
do
    echo ${i}
    for step in step3125 step6250 step9375 step12500 step15625 step18750 step21875 step25000 step28125 step31250       	
    	do
		echo ${i}${step}
		python3 upload_branch.py ${i} ${step}
	done
done
