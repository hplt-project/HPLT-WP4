checkpoints_path=${1}
out_path=${2}
for checkpoint in $checkpoints_path/*.bin
    do
        echo $checkpoint
        lang=$(awk -v var="$checkpoint" 'BEGIN { print substr(var,length(var)-16, 8) }')
        echo  $lang
        cp -r huggingface_prototype/ $out_path/$lang/
        cp $checkpoint $out_path/$lang/model.bin
    done