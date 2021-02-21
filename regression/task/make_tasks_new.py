import torch
import numpy as np
import random
from functools import partial

# from .regression_1d import sample_sin_fnc, sample_linear_fnc
from .image_reconstruction_new import sample_fnc_helper, sample_mnist_img_fnc, sample_fmnist_img_fnc
# from .image_reconstruction import sample_celeba_img_fnc, sample_cifar10_img_fnc, create_hier_imagenet_supertasks
from .LQR import sample_LQR_LV2, sample_LQR_LV1, sample_LQR_LV0 

from pdb import set_trace



##############################################################################
class Task_sampler():
    def __init__(self, task_fnc, param_fnc = None): 
        if isinstance(task_fnc, dict):
            assert param_fnc is None
            param_fnc = list(task_fnc.keys())
            task_fnc = task_fnc.get
        self.task_fnc  = task_fnc    # function that takes task label and returns actual tasks
        self.param_fnc = param_fnc   # 
        self.params = None

    def pre_sample(self, k_batches):  # pre-sample k_batches of params  
        self.params = sample_shuffle_split(self.param_fnc, k_batches)
        for type_ in ['test', 'valid']:
            if len(self.params[type_]) == 0:
                self.params[type_] = self.params['train']  # duplicate train and test params.

    def get_data(self, sample_type): 
        if self.params is None: 
            error()
        params = self.params[sample_type]
        return params, batch_wrapper(self.task_fnc)(params) 

#####################
def Partial_Task_sampler(task_fnc, input_fnc):
    def fnc_(params):
        return Task_sampler(task_fnc = partial(task_fnc, params), param_fnc = input_fnc)
    return fnc_

##########################
def make_dict(keys, vals):
    return {k:v for (k,v) in zip(keys, vals)}

def sample_shuffle_split(input_, k_batches: dict):
    types = ['train', 'test', 'valid']
    set_trace()
    if k_batches is None and hasattr(input, '__len__'): 
        k_batches_list = [len(input_),0,0]
    else:
        assert  types == list(k_batches.keys()) 
        k_batches_list = list(k_batches.values())
    assert len(k_batches_list) == 3

    idx = np.cumsum([0]+k_batches_list)
    total_batch = idx[-1]
    if callable(input_):                         # param generating function
        input_ = input_(batch = total_batch)     # list of sampled params
    assert isinstance(input_, (list, np.ndarray, torch.FloatTensor)) 

    if isinstance(input_, list): 
        random.shuffle(input_)
        return make_dict(types, [input_[idx[i]:idx[i+1]] for i in range(len(idx)-1)])

    elif isinstance(input_, torch.FloatTensor):
        rnd_idx = torch.randperm(input_.shape[0])
        input_ = input_[rnd_idx].view(input_.shape)
        return make_dict(types, [input_[idx[i]:idx[i+1], :] for i in range(len(idx)-1)])

    elif isinstance(input_, np.ndarray):
        raise NotImplementedError
    else:
        raise NotImplementedError


##############################################################

def batch_wrapper(fnc):
    def help_fnc(batch, *args):
        if isinstance(batch, (np.ndarray, torch.FloatTensor)):
            return fnc(batch, *args)
        elif isinstance(batch, list):
            return [fnc(b, *args) for b in batch]
        elif isinstance(batch, (int, np.int64, np.int32)):
            return [fnc(*args) for _ in range(batch)]
        else:
            print(type(batch), batch)
            error()
    return help_fnc

###########################################
#  1-D regression task
from .regression_1d import input_gen_1d, sine_params, line_params, sine1_fnc, line1_fnc

########
# Level0
sine0_task = Partial_Task_sampler(sine1_fnc, input_gen_1d)
line0_task = Partial_Task_sampler(line1_fnc, input_gen_1d)
######## 
# Level1
sine_param_gen = batch_wrapper(sine_params)
line_param_gen = batch_wrapper(line_params) 
sine1_task = Task_sampler(task_fnc = sine0_task, param_fnc = sine_param_gen)
line1_task = Task_sampler(task_fnc = line0_task, param_fnc = line_param_gen)
######## 
# Level2
sine_line2_dict = {'sine': sine1_task, 'line': line1_task}
sine_line2_task = Task_sampler(task_fnc = sine_line2_dict, param_fnc = None) 


###########################################
#  2-D regression (Image -reconstruction ) task

##########
# sample_label_mnist = Task_sampler(tasks = list(range(10)), k_batches = [7,3,0], task_gen_fnc = sample_mnist_img_fnc).sample
# sample_label_fmnist = Task_sampler(tasks = list(range(10)), k_batches = [7,3,0], task_gen_fnc = sample_fmnist_img_fnc).sample
# sample_label_mnist_fmnist = Task_sampler(tasks = [sample_label_mnist, sample_label_fmnist]).sample
# # sample_label_mnist_fmnist = Task_sampler(tasks = ['mnist', 'fmnist'], task_gen_fnc = sample_fnc_helper).sample



get_task_dict={
    'sine':   sine1_task,
    'linear': line1_task,
    'sine+linear': sine_line2_task,
#     
    'LQR_lv2': sample_LQR_LV2,
    'LQR_lv1': sample_LQR_LV1,
    'LQR_lv0': (sample_LQR_LV0, None),
#
#     'mnist'  : sample_label_mnist, 
#     'fmnist' : sample_label_fmnist, 
#     'mnist+fmnist' : sample_label_mnist_fmnist,
}

