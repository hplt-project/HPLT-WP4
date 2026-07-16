for checkpoint in /cluster/work/projects/nn9851k/mariiaf/hplt/ud_checkpoints/*.bin
    do
        echo $checkpoint
        lang=$(awk -v var="$checkpoint" 'BEGIN { print substr(var,length(var)-16, 8) }')
        echo  $lang
        cp -r huggingface_prototype/ /cluster/work/projects/nn9851k/mariiaf/hplt_ud_hf/$lang/
        cp $checkpoint /cluster/work/projects/nn9851k/mariiaf/hplt_ud_hf/$lang/model.bin
    done