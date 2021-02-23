from hierarchical_task import Task_sampler, Partial_sampler, batch_wrapper



###########################################
#  1D regression task

from task.regression_1d import input_fnc_1d, sine_params, line_params, sine1_fnc, line1_fnc, sine_linear2_fnc

sine0_task = Partial_sampler(sine1_fnc, input_fnc_1d)
sine1_task = Task_sampler(task_fnc = sine0_task, param_fnc = batch_wrapper(sine_params))

line0_task = Partial_sampler(line1_fnc, input_fnc_1d)
line1_task = Task_sampler(task_fnc = line0_task, param_fnc = batch_wrapper(line_params) )

sine_line2_task = Task_sampler(task_fnc = {'sine': sine1_task, 'line': line1_task}, param_fnc = None) 

####################

sine_linear0_task = Partial_sampler(sine_linear2_fnc, input_fnc_1d)
sine_linear1_task = Partial_sampler(sine_linear0_task, batch_wrapper(sine_params))
sine_linear2_task = Task_sampler(sine_linear1_task, batch_wrapper(line_params))


###########################################
#  2D regression (Image -reconstruction ) task

from task.image_reconstruction import img_reconst_task_gen

mnist_lv2_fnc, in_fnc_2d_coord = img_reconst_task_gen('mnist')

mnist0_task = Partial_sampler(mnist_lv2_fnc, in_fnc_2d_coord)  # lv0 : pixel
mnist1_task = Partial_sampler(mnist0_task, None)               # lv1 : choose image 
mnist2_task = Task_sampler(mnist1_task, None)                  # lv2 : choose label 

# mnist1_task(label = 1) : level1 task with digit =1 only
# mnist1_task(label = None)  : level1 task with all digits

fmnist_lv2_fnc, in_fnc_2d_coord = img_reconst_task_gen('fmnist')

fmnist0_task = Partial_sampler(fmnist_lv2_fnc, in_fnc_2d_coord)  # lv0 : pixel
fmnist1_task = Partial_sampler(fmnist0_task, None)               # lv1 : choose image 
fmnist2_task = Task_sampler(fmnist1_task, None)                  # lv2 : choose label 

mnist_fmnist3_task = Task_sampler({'mnist_lv2': mnist2_task, 'fmnist_lv2': fmnist2_task}, None)

################################################
# LQR tasks -  to be fixed
from task.LQR import sample_LQR_LV2, sample_LQR_LV1, sample_LQR_LV0 

task_dict={
    'sine_lv1':      sine1_task,
    'line_lv1':      line1_task,
    'sine+line_lv2': sine_line2_task,
#     'sine+line_lv1': sine_line2_task(None),
    'sine_linear_lv2': sine_linear2_task,
#
#     'mnist_lv0'  : mnist2_task(None, None), 
    'mnist_lv1'  : mnist1_task(None), 
    'mnist_lv2'  : mnist2_task, 
    'fmnist_lv1' : fmnist1_task(None), 
    'fmnist_lv2' : fmnist2_task, 
    'mnist+fmnist_lv3' : mnist_fmnist3_task,
#     'mnist+fmnist_lv2' : mnist_fmnist3_task(None),
#     
    'LQR_lv2': sample_LQR_LV2,
    'LQR_lv1': sample_LQR_LV1,
    'LQR_lv0': (sample_LQR_LV0, None),
}


# lv3: dataset f( .  , rnd, type)    -> dataset    (e.g. MNIST vs fMNIST)
# lv2: label   f(dataset, rnd, type) -> label                -> Loss( theta | dataset )           -> averaged over labels
# lv1: image   f(label, rnd, type)   -> image                -> Loss( ctx1 | label | theta)       -> averaged over image
# lv0: pixel   f(image, rnd, type)   -> xy, pixel (target)   -> Loss( ctx0 | image | ctx1, theta) -> averaged over xy E_xy[L(xy)]


