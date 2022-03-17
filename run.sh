#!/bin/bash 
# Go to the dir of the scirpt firstly before running all commands 
# The script should be in the root dir of the 2DMotion project
path="$(cd "$(dirname $0)";pwd)"
cd ${path}

# conda run -n 2dmotion python eval.py -ss finetuning -n 2dmotion --test
conda run -n 2dmotion python eval.py -ss finetuning -ms ocv0_rgb ocv0_depth ocv0_rgbd -n 2dmotion --test --opts most_frequent_pred origin_NOC
conda run -n 2dmotion python eval.py -ss finetuning -ms ocv0_rgb ocv0_depth ocv0_rgbd -n 2dmotion --test --opts most_frequent_pred random_NOC


# conda run -n 2dmotion python eval.py -ss finetuning -ms cc_rgb cc_depth cc_rgbd -n 2dmotion --opts random_baseline gtbbx gtcat

