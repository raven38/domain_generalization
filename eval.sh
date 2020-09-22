#!/bin/bash

domain=("photo" "art" "cartoon" "sketch")

max=$((${#domain[@]}-1))
for j in `seq 0 $max`
do    
    dir_name="/home/results/${domain[j]}"
    snapshot="${dir_name}/model_100"
    echo $dir_name
    python pacs_main.py  --data-root /kfold/  --save-path $dir_name  --snapshot ${snapshot} --mode eval --exp-num $j --batch-size 256 
done
