#!/bin/bash

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=hml
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo " Run started at:- "
date

python3 main_huh.py \
--task sine+line_lv2 --n_contexts 4 0 --n_iters 3 0 5000 --for_iters 3 0 1 \
--architecture 1 40 40 1 --lrs 0.03 0.03 0.001 --test_intervals 1 0 1 \
--k_batch_train 10 10000 2 --n_batch_train 10 50 2 --k_batch_test 101 10000 2 --n_batch_test 101 1 1 \
--log-name sine+line_lv1 --log_level_loss True True True --log_level_ctx True True False --task_separate_levels True True True \
--private 

echo "Run completed at:- "
date
