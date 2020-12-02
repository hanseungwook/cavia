#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Append PYTHONPATH that points to gym-minigrid
export PYTHONPATH=/home/dongki/research/lids/git/gym-minigrid:$PYTHONPATH

# Begin experiment
python3 main.py \
--prefix "Empty-Empty"
