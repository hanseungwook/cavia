#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Append PYTHONPATH that points to gym-minigrid
export PYTHONPATH=/home/dongki/research/lids/git/gym-minigrid:$PYTHONPATH

# Begin experiment
cd $DIR
for seed in {1..1}
do
    python3.6 simple_meta_example.py
done
