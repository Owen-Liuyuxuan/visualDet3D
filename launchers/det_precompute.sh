#!/bin/bash
set -e
if [[ "$2" == "" ]];then
    echo -e "--------------------Precomputation for object detection script------------------"
    echo -e "Two arguments are needed. Usage: \n"
    echo -e "   ./det_precompute.sh <ConfigPath(str)> <SPLIT(train/test)>\n"
    echo -e "If SPLIT==train, we will directly launch imdb_precompute_3d.py,\notherwise we will launch imdb_precompute_test.py\n"
    echo -e "exiting"
    echo -e "------------------------------------------------------------------"
    exit 1
fi
CONFIG_PATH=$1
SPLIT=$2

if [ $SPLIT == "train" ]; then 
    echo -e "Precomputation for the training/validation split"
    python3 scripts/imdb_precompute_3d.py --config=$CONFIG_PATH
else
    echo -e "Precomputation for the test split"
    python3 scripts/imdb_precompute_test.py --config=$CONFIG_PATH
fi
