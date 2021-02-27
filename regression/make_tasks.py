from hierarchical_task import Task_sampler, Partial_sampler, batch_wrapper

import random 

from pdb import set_trace

###############################################
# make_composite_task

def combine_names(names):
    return "".join([name+'+' for name in names])[:-1]

def make_composite_task(task_dict):
    names = list(task_dict.keys())
    def high_level_params():
        return random.choice(names)

    def high_level_fnc(name):
        return task_dict[name]  

    return (high_level_fnc, batch_wrapper(high_level_params))

###########################################
#  1D regression task

from task.regression_1d import sine_gen, line_gen, sine_linear_gen


###########################################
#  2D regression (Image -reconstruction ) task

from task.image_reconstruction import img_reconst_gen

################################################
# LQR tasks 
# from task.LQR import LQR2_fnc, LQR_param_lv0, LQR_param_lv1, LQR_param_lv2  #sample_LQR_LV2, sample_LQR_LV1, sample_LQR_LV0 

################################################

task_dict={
    'sine':      sine_gen,
    'line':      line_gen,
    'sine_linear': sine_linear_gen,
#
    'mnist'  : img_reconst_gen('mnist'),
    'fmnist' : img_reconst_gen('fmnist'),
    'cifar10': img_reconst_gen('cifar10'),
    # 
    # 'LQR_lv2': LQR2_gen,
}


def get_task(name_):
    # assert isinstance(names, (list,tuple)), "task names should be a list or tuple."
    names = name_.split("+")  # split into a list of tasks
    task = make_composite_task({name:task_dict[name]() for name in names})
    if len(names) == 1: 
        return task
    else: 
        # name = combine_names(names)
        supertask = make_composite_task({name_:task})  # 
        return supertask


# lv3: dataset f( .  , rnd, type)    -> dataset    (e.g. MNIST vs fMNIST)
# lv2: label   f(dataset, rnd, type) -> label                -> Loss( theta | dataset )           -> averaged over labels
# lv1: image   f(label, rnd, type)   -> image                -> Loss( ctx1 | label | theta)       -> averaged over image
# lv0: pixel   f(image, rnd, type)   -> xy, pixel (target)   -> Loss( ctx0 | image | ctx1, theta) -> averaged over xy E_xy[L(xy)]


