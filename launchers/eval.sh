#!/bin/bash
set -e
if [[ $3 == "" ]];then
    echo -e "--------------------Distribution training script------------------"
    echo -e "Four arguments are needed. Usage: \n"
    echo -e "   ./eval.sh <ConfigPath> <GPU[int]> <CheckPointPath [str]> <Split [Optinal[str]:validation/test> \n"
    echo -e "launch script"
    echo -e "exiting"
    echo -e "------------------------------------------------------------------"
    exit 1
fi
CONFIG_PATH=$1
GPU=$2
CKPT_PATH=$3
SPLIT=$4
if [[ "$SPLIT" == '' ]]; then
    echo -e "SPLIT not set. validation by default"
    SPLIT="validation"
fi

CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval.py --config=$CONFIG_PATH --gpu=0 --checkpoint_path=$CKPT_PATH --split_to_test=$SPLIT 

