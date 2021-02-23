from dataset import DataLoader, Meta_DataLoader, Dataset_helper #, Hierarchical_Task


# def empty_wrapper(input_gen, target_gen ):
#     def helper_fnc(sample_type, total_batch):  # ignore sample_type
#         x  = input_gen(total_batch)  # Huh : added sample_type as input
#         target = target_gen(x)
#         return x, target
#     return helper_fnc

import numpy as np
import random
from pdb import set_trace
##########################



##########################

         
# class Task_sampler2():
#     def __init__(self, task_fnc, in_fnc = None, k_batches: list = None):    
#         if isinstance(task_fnc, dict):
#             assert in_fnc is None
#             in_fnc = list(task_fnc.keys())
#             task_fnc = task_fnc.get
#         self.task_fnc = task_fnc  # function that takes task label and returns actual tasks
        
#         in_fnc = sample_shuffle_split(in_fnc, k_batches)
#         self.params={k:v for (k,v) in zip(['train', 'test', 'valid'], in_fnc)}
        
#         if len(self.params['test']) == 0:
#             self.params['test'] = self.params['train']  # duplicate train and test params.
            

#     def get_data(self, sample_type): 
#         params = self.params[sample_type]
#         return params, batch_wrapper(self.task_fnc)(params) 


class Task_package():
    def __init__(in_fnc, task_fnc):
        self.in_fnc = in_fnc
        self.task_fnc = task_fnc
        
    def partial(self, param):
        in_ = partial(self.in_fnc, param)
        task_ = partial(self.task_fnc, param)
        return Task_package(in_, task_)
    
    def sample(self, batch_or_type):
        x = self.in_fnc(batch_or_type)
        y = self.task_fnc(x)
        return x, y
    
    def go_lower(self, ):
        _, param = self.sample()
        return self.partial(self, param)
    
###########################################
def Meta_model0():
    def __init__(base_model):
        self.base_model = base_model
    
    def forward(task):
        lower_tasks = task.lower()
        if lower_tasks is not None: #  level == 0:
            loss = get_average_loss(self, lower_tasks)
        else:
            loss = self.base_model(task)
        return loss
    

###########################################
def Meta_model1():
    def __init__(self, base_model):
        self.base_model = base_model
    
    def evaluate(self, task_):
        lower_tasks = task_.lower()
        if lower_tasks is not None: #  level == 0:
            loss = get_average_loss(self, lower_tasks)
        else:
            loss = self.base_model(task_)
        return loss
    
    def forward(task):
        optimize(self, task('train'))  #  inner-loop optim
        test_loss = self.evaluate(task('test'))
        return test_loss
    
    
def get_average_loss(model, tasks):
    loss = []
    for t_ in tasks:
        l = model(t_)
        loss.append(l)
    return  torch.stack(loss).mean() 


###################################
       
# import torch

# input_1d_range = [-5, 5]

# def input_fnc_1d(batch, grid=False):         # Full inputs over the whole regression input range
#     if grid:
#         return torch.linspace(input_1d_range[0], input_1d_range[1], steps=batch).unsqueeze(1)
#     else:
#         return torch.rand(batch, 1) * (input_1d_range[1] - input_1d_range[0]) + input_1d_range[0]


# def sine_params():       # Sample n_batch number of parameters
#     amplitude = np.random.uniform(0.1, 5.)
#     phase = np.random.uniform(0., np.pi)
#     return amplitude, phase


# def line_params():
#     slope = np.random.uniform(-3., 3.)
#     bias = np.random.uniform(-3., 3.)
#     return slope, bias


# sine1_param_gen = batch_wrapper(sine_params)
# line_param_gen = batch_wrapper(line_params) 
