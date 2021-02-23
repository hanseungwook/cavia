import torch
import numpy as np
import random
from functools import partial

from dataset import Meta_DataLoader, get_Dataset
from torch.utils.data import DataLoader #, Dataset, Subset

##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]

class Hierarchical_Task():   
    # Top-down generation of task hierarchy.
    
    def __init__(self, task_sampler, batch_dict, idx=0): 
        # level = len(batch_dict[0]) - 1          # print(self.level, total_batch_dict)
        self.dataloaders = get_dataloader_dict(task_sampler, batch_dict, idx)
        
#         if print_hierarhicial_task:
#             print('level', level, 'batch', batch_dict[0][-1]['train'])
#             print('task', task_sampler)

#         if print_task_loader and level>0:
#             print('Task_loader Level', level, 'task', task_sampler)

    def load(self, sample_type):   
        loader = self.dataloaders[sample_type] or self.dataloaders['train']  # duplicate 'train' loader, if 'test'/'valid' loader == None
        if sample_type == 'train':
            return loader                            # return dataloader
        else: 
            return next(iter(loader))                # return one iter from dataloader


######################
# get_dataloader_dict

def get_dataloader_dict(task_sampler, batch_dict, idx):    
    batch_dict_next = (batch_dict[0][:-1], batch_dict[1][:-1])
    total_batch, mini_batch = batch_dict[0][-1], batch_dict[1][-1]           # mini_batch: mini batch # of samples

    def get_dataloader(sample_type, mini_batch_):     #     sample_type = 'train' or 'test'  
        input_params, target_samplers = task_sampler.get_data(sample_type)
        if len(input_params) == 0:     # No dataloader for empty dataset sampler
            return None
        else:
            if isinstance(target_samplers, (np.ndarray, torch.FloatTensor)):       # if level == 0:
                # input_data, target = input_params, target_samplers   # returns tensors
                return DataLoader(get_Dataset(input_params, target_samplers), batch_size=mini_batch_, shuffle=(sample_type == 'train')) 
            else:
                data = [Hierarchical_Task(target_sampler, batch_dict_next, idx_) for (idx_, target_sampler) in enumerate(target_samplers)]
                return Meta_DataLoader(get_Dataset(input_params, data), batch_size=mini_batch_, idx=idx)   # , name=str(task_sampler)

    task_sampler.pre_sample(total_batch)  # pre-sampling 
    loader_dict = {key: get_dataloader(key, mini_batch[key]) for key in ['train', 'test', 'valid']}
    return loader_dict



#####################
# Task_sampler

class Task_sampler():
    def __init__(self, task_fnc, param_fnc = None): 
        if isinstance(task_fnc, dict):
            assert param_fnc is None
            param_fnc = list(task_fnc.keys())
            task_fnc = task_fnc.get
        self.task_fnc  = task_fnc    # function that takes task label and returns actual tasks
        self.param_fnc = param_fnc   # 
        self.params = None

    def pre_sample(self, k_batches: dict):              # pre-sample k_batches of params  
        self.params = sample_shuffle_split(self.param_fnc, k_batches)
        # for type_ in ['test', 'valid']:
        #     if len(self.params[type_]) == 0:
        #         self.params[type_] = self.params['train']   # duplicate train and test params.

    def get_data(self, sample_type): 
        if self.params is None: 
            error()
        params = self.params[sample_type]
        return params, batch_wrapper(self.task_fnc)(params) 

#     def __str__(self):
#         return self.param_fnc.__name__ if hasattr(self.param_fnc, '__name__')  else 'None' # self.task_fnc.__name__ #self.params

#####################
# Partial_sampler

def Partial_sampler(task_fnc, input_fnc): #, which = 'task' ):
    def fnc_(*params):
        return Task_sampler(task_fnc = partial(task_fnc, *params), param_fnc = input_fnc)
    return fnc_


########################
# sample_shuffle_split

def sample_shuffle_split(input, k_batches: dict):

    keys = ['train', 'test', 'valid']

    def make_dict(vals):
        return {k:v for (k,v) in zip(keys, vals)}

    def make_list(x):
        if x is None and hasattr(input, '__len__'): 
            x_list = [len(input),0,0]
        else:
            assert  keys == list(x.keys()), "wrong keys for x dict"
            x_list = list(x.values())

        x_list = [a if a is not None else 0 for a in x_list]        # replace None with 0
        assert len(x_list) == 3, "wrong number of x_list items"
        return x_list

    def sample(input, total_batch):
        if input is None:
            return list(range(total_batch))
        elif callable(input):
            return input(total_batch) 
        else:
            return input

    ################
    # main code
    k_list = make_list(k_batches)
    inputs = sample(input, sum(k_list))     # list of sampled params
    assert isinstance(inputs, (list, torch.FloatTensor)), "wrong type of inputs"  # , np.ndarray
        
    ###  shuffle and split and make_dict 
    idx = np.cumsum([0]+k_list);   # np.cumsum(filter(None, [0]+k_list))  #  # total_batch = idx[-1]

    if isinstance(inputs, list): 
        random.shuffle(inputs)
        return make_dict([inputs[idx[i]:idx[i+1]] for i in range(len(idx)-1)])

    elif isinstance(inputs, torch.FloatTensor):
        rnd_idx = torch.randperm(inputs.shape[0])
        inputs = inputs[rnd_idx].view(inputs.shape)  # shuffled tensor
        return make_dict([inputs[idx[i]:idx[i+1], :] for i in range(len(idx)-1)])

    # else:
    #     error()


##################
# batch_wrapper:  make a batch version of a function

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
