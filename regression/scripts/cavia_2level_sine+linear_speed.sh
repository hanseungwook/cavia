#!/bin/bash

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=hml
CONDA_ROOT=/disk_c/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo " Run started at:- "
date
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES="" python3 main_huh.py \
--task sine+line --n_contexts 2 2 --max_iters 3 3 100 --for_iters 3 3 1 \
--architecture 1 40 40 1 --lrs 0.03 0.03 0.001 --test_intervals 4 4 2 \
--k_train 400 10000 2 --n_train 100 10 2 \
--log-name sine+line_lv2_cpu_100_10_2 --log_loss_levels 0 1 2 --task_separate_levels 0 1 2 \
--print_levels 2 --private \

END_TIME=$(date +%s)
echo "Time: $(($END_TIME - $START_TIME))"
echo "Run completed at:- "
date
