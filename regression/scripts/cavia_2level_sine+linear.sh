#!/bin/bash

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=hml
CONDA_ROOT=/disk_c/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo " Run started at:- "
date

CUDA_VISIBLE_DEVICES=1 python3 main_huh.py \
--task sine+line_lv2 --n_contexts 2 2 --n_iters 3 3 2000 --for_iters 3 3 1 \
--architecture 1 40 40 1 --lrs 0.03 0.03 0.001 --test_intervals 1 1 1 \
--k_batch_train 10 10000 2 --n_batch_train 10 50 2 --k_batch_test 101 10000 2 --n_batch_test 101 50 2 \
--log-name sine+line_lv2 --log_loss_levels 0 1 2 --log_ctx_levels 0 1 --task_separate_levels 0 1 2 \
--private \

echo "Run completed at:- "
date
