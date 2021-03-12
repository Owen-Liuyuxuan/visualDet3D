#!/bin/bash
set -e
if [[ "$2" == "" ]];then
    echo -e "--------------------Disparity Ground Truth Precompute script------------------"
    echo -e "Two arguments are needed. Usage: \n"
    echo -e "   ./disparity_precompute.sh <ConfigPath> <IsUsingPointCloud>\n"
    echo -e "exiting"
    echo -e "------------------------------------------------------------------"
    exit 1
fi
CONFIG_PATH=$1
IS_PC=$2


python3 scripts/disparity_compute.py --config=$CONFIG_PATH --use_point_cloud=$IS_PC
