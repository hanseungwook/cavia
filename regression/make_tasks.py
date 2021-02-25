from hierarchical_task import Task_sampler, Partial_sampler, batch_wrapper

import random 

from pdb import set_trace

###############################################
# make_composite_task

def make_composite_task(task_dict):
    names = list(task_dict.keys())
    def high_level_params():
        return random.choice(names)

    def high_level_fnc(name):
        return task_dict[name]  

    return high_level_fnc, batch_wrapper(high_level_params)

###########################################
#  1D regression task

from task.regression_1d import sine1_task, line1_task, sine_linear2_task


###########################################
#  2D regression (Image -reconstruction ) task

from task.image_reconstruction import img_reconst_task_gen

cifar10_task = img_reconst_task_gen('cifar10')

mnist2_task = img_reconst_task_gen('mnist')
fmnist2_task = img_reconst_task_gen('fmnist')
mnist_fmnist3_task = make_composite_task({'mnist_lv2': mnist2_task, 'fmnist_lv2': fmnist2_task})

################################################
# LQR tasks 
# from task.LQR import LQR2_fnc, LQR_param_lv0, LQR_param_lv1, LQR_param_lv2  #sample_LQR_LV2, sample_LQR_LV1, sample_LQR_LV0 

################################################

task_dict={
    'sine_lv1':      sine1_task,
    'line_lv1':      line1_task,
    'sine+line_lv2': make_composite_task({'sine': sine1_task, 'line': line1_task}),
    'sine_linear_lv2': sine_linear2_task,
#
    'mnist_lv2'  : mnist2_task,
    'fmnist_lv2' : fmnist2_task,
    'mnist+fmnist_lv3' : make_composite_task({'mnist_lv2': mnist2_task, 'fmnist_lv2': fmnist2_task}),
# 
    'cifar10_lv2'  : cifar10_task,
    # 'LQR_lv2': LQR2_task,
}

def get_task(name):
    return make_composite_task({'root': task_dict[name]})

# lv3: dataset f( .  , rnd, type)    -> dataset    (e.g. MNIST vs fMNIST)
# lv2: label   f(dataset, rnd, type) -> label                -> Loss( theta | dataset )           -> averaged over labels
# lv1: image   f(label, rnd, type)   -> image                -> Loss( ctx1 | label | theta)       -> averaged over image
# lv0: pixel   f(image, rnd, type)   -> xy, pixel (target)   -> Loss( ctx0 | image | ctx1, theta) -> averaged over xy E_xy[L(xy)]


