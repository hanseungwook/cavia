import torch
import numpy as np
import random

from dataset import Meta_DataLoader, get_Dataset
from torch.utils.data import DataLoader #, Dataset, Subset
from utils import batch_wrapper

from pdb import set_trace

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
    
    def __init__(self, task, batch_dict):
        self.task = task
        self.dataloaders = get_dataloader_dict(task, batch_dict) 

    def load(self, sample_type):   
        loader = self.dataloaders[sample_type] or self.dataloaders['train']  # duplicate 'train' loader, if 'test'/'valid' loader == None
        if sample_type == 'train':
            return loader                            # return dataloader
        else: 
            return next(iter(loader))                # return one iter from dataloader


######################
# get_dataloader_dict

def get_dataloader_dict(task, batch_dict):
    batch_dict_next = (batch_dict[0][:-1], batch_dict[1][:-1])
    total_batch, mini_batch = batch_dict[0][-1], batch_dict[1][-1]           # mini_batch: mini batch # of samples

    sampler = Task_sampler(*task, total_batch) 

    def get_dataloader(sample_type, mini_batch_):   
        input_params, task_list_next = sampler.get_data(sample_type)

        if len(input_params) == 0:     # No dataloader for empty dataset sampler
            return None
        else:
            if len(batch_dict_next[0]) > 0:  #high-level
                task_list = [Hierarchical_Task(task_next, batch_dict_next) for task_next in task_list_next]
                return Meta_DataLoader(get_Dataset(input_params, task_list), batch_size=mini_batch_)
            else:                           # level == 0: 
                inputs, targets = input_params, task_list_next
                return DataLoader(get_Dataset(inputs, targets), batch_size=mini_batch_, shuffle=(sample_type == 'train')) 

    loader_dict = {key: get_dataloader(key, mini_batch[key]) for key in ['train', 'test', 'valid']}
    return loader_dict



#####################
# Task_sampler

class Task_sampler():
    def __init__(self, task_fnc, param_fnc, k_batches): 
        if isinstance(task_fnc, dict):
            assert param_fnc is None
            param_fnc = list(task_fnc.keys())
            task_fnc = task_fnc.get
        self.task_fnc  = task_fnc    # function that takes task label and returns actual tasks
        self.param_fnc = param_fnc  
        self.params = sample_shuffle_split(param_fnc, k_batches)       # pre-sample k_batches of params  

    def get_data(self, sample_type): 
        assert self.params is not None, "run pre_sample first"
        params = self.params[sample_type]
        return params, batch_wrapper(self.task_fnc)(params) 

########################
# sample_shuffle_split

def sample_shuffle_split(input, k_batches: dict):
    keys = ['train', 'test', 'valid']

    def make_dict(vals):
        return {k:v for (k,v) in zip(keys, vals)}

    def make_list(x):
        if isinstance(x, dict):
            x_list = list(x.values());    assert  keys == list(x.keys()), "wrong keys for x dict"
            x_list = [a if a is not None else 0 for a in x_list]        # replace None with 0
        else: 
            x_list = [len(input),0,0];    assert  x is None and hasattr(input, '__len__')
        return x_list

    ################
    # main code
    k_list = make_list(k_batches); 
    total_batch = sum(k_list)
    # set_trace()
    if input is None:
        input = list(range(total_batch))
    inputs = input(total_batch) if callable(input) else random.sample(input, total_batch)   #  # inputs = sample(input, sum(k_list))     # list of sampled params
    # assert isinstance(inputs, (list, torch.FloatTensor)), "wrong type of inputs"  # , np.ndarray
    
    ###  shuffle and split and make_dict 
    idx = np.cumsum([0]+k_list);   # np.cumsum(filter(None, [0]+k_list))  #  # total_batch = idx[-1]

    return make_dict([inputs[idx[i]:idx[i+1]] for i in range(len(idx)-1)])
