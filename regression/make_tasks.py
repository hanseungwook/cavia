from hierarchical_task import Task_sampler, batch_wrapper
import random 

from task.regression_1d import sine_gen, line_gen, sine_linear_gen    #  1D regression task
from task.image_reconstruction import img_reconst_gen   #  2D regression (Image -reconstruction ) task
from task.LQR import LQR_gen                            # LQR task

from pdb import set_trace

###############################################
# make_composite_task

# def combine_names(names):
#     return "".join([name+'+' for name in names])[:-1]

def make_composite_task(task_dict):
    names = list(task_dict.keys())
    def high_level_params():
        return random.choice(names)

    def high_level_fnc(name):
        return task_dict[name]  

    return (high_level_fnc, batch_wrapper(high_level_params))

###########################################

task_dict={
    'sine':      sine_gen,
    'line':      line_gen,
    'sine_linear': sine_linear_gen,
#
    'mnist'  : img_reconst_gen('mnist'),
    'fmnist' : img_reconst_gen('fmnist'),
    'cifar10': img_reconst_gen('cifar10'),
    # 
    'LQR': LQR_gen,
}


def get_task(name_str: str):
    name_list = name_str.split("+")  # split into a list of task names
    task = make_composite_task({name:task_dict[name]() for name in name_list})
    if len(name_list) == 1: 
        return task
    else: 
        return make_composite_task({name_str:task})  # supertask

