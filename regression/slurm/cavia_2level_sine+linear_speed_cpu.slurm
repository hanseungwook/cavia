#!/bin/bash
###  replace the .py file in the horovodrun call with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err

#SBATCH -J speed_cpu
#SBATCH -o speed_cpu_%j.out
#SBATCH -e speed_cpu_%j.err

#SBATCH --mail-user=seungwook.han@ibm.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=500g
#SBATCH --time=24:00:00
#SBATCH -p sched_system_all 

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=wmlce-ea
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date


## Horovod execution
###  replace the .py with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err 
n_train_0=(100 200 400)
n_train_1=(10 20 40)
n_train_2=(2 4 8)

for i in "${n_train_0[@]}"; do
    for j in "${n_train_1[@]}"; do
        for k in "${n_train_2[@]}"; do
            for t in {1..10..1}; do
                START_TIME=$(date +%s)
                CUDA_VISIBLE_DEVICES="" python3 main_huh.py \
                --task sine+line --n_contexts 2 2 --max_iters 3 3 20 --for_iters 3 3 1 \
                --architecture 1 40 40 1 --lrs 0.03 0.03 0.001 --test_intervals 4 4 2 \
                --k_train 400 10000 2 --n_train "$i" "$j" "$k" \
                --log-name sine+line_lv2_cpu_100_10_2 --log_loss_levels 0 1 2 --task_separate_levels 0 1 2 \
                --print_levels 2 --private \
                END_TIME=$(date +%s)
                echo "[$i $j $k] Time: $(($END_TIME - $START_TIME))"
            done
        done
    done
done

echo "Run completed at:- "
date
