#!/bin/bash
set -e
if [[ $3 == "" ]];then
    echo -e "--------------------(Un)Distributed training script------------------"
    echo -e "Three arguments are needed. Usage: \n"
    echo -e "   ./train.sh <ConfigPath> <GPUS_SEP_BY_COMMA_NO_SPACE [int, int, ...]> <EXPERIMENT_NAME [str]>\n"
    echo -e "If only one GPU is assigned, we will directly launch train.py,\notherwise we will launch torch.distributed.launch\n"
    echo -e "exiting"
    echo -e "------------------------------------------------------------------"
    exit 1
fi
CONFIG_PATH=$1
GPUS=$2
EXP_NAME=$3
NUM_GPUS=$(($(echo $GPUS | grep -o ',' | wc -l) + 1)) # count number of ',' and plus one

if [ $NUM_GPUS == 1 ]; then 
    echo -e "Nnumber of GPUs being 1, will directly launch:\n\t python3 train --experiment_name=$EXP_NAME"
    CUDA_VISIBLE_DEVICES=$GPUS python3 scripts/train.py --config=$CONFIG_PATH --experiment_name=$EXP_NAME 
else
    echo -e "Distributed Training on GPU $GPUS, total number of gpus is $NUM_GPUS\n"
    CUDA_VISIBLE_DEVICES=$GPUS python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/train.py --config=$CONFIG_PATH --experiment_name=$EXP_NAME --world_size=$NUM_GPUS
fi
