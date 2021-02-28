#!/bin/bash

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=hml
CONDA_ROOT=/disk_c/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo " Run started at:- "
date

CUDA_VISIBLE_DEVICES=0 python3 main_huh.py \
--task sine+line_lv2 --n_contexts 4 0 --max_iters 3 0 2000 --for_iters 3 1 1 \
--architecture 1 40 40 1 --lrs 0.03 0.0 0.001 --test_intervals 1 0 1 \
--k_train 10 10000 2 --n_train 10 50 2 --k_test 101 10000 2 --n_test 101 50 2 \
--log-name sine+line --log_loss_levels 0 1 2 --log_ctx_levels 0 1 --task_separate_levels 0 1 2 \
--print_levels 2 --private \

echo "Run completed at:- "
date
