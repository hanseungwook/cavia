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
    
    def __init__(self, task_gen, total_batches, mini_batches):
        total_batch, mini_batch = total_batches[-1], mini_batches[-1]     # for current level  
        batch_dicts_next = (total_batches[:-1], mini_batches[:-1])        # for next level 

        sampler = Task_sampler(task_gen, total_batch) 
        self.dataloaders = get_dataloader_dict(sampler, mini_batch, batch_dicts_next) 

    def load(self, sample_type):   
        return self.dataloaders[sample_type] or self.dataloaders['train']  # duplicate 'train' loader, if 'test'/'valid' loader == None
        
#####################
# get_dataloader


def get_dataloader_dict(sampler, mini_batch, batch_dicts_next):

    def get_dataloader(sample_type, mini_batch):   
        param_samples, task_samples = sampler.data(sample_type)

        if len(param_samples) == 0:     # No dataloader for empty data samples
            return None
        else:
            if len(batch_dicts_next[0]) == 0:    # level0:
                inputs, targets = param_samples, task_samples  # re-naming
                return DataLoader(get_Dataset(inputs, targets), batch_size=mini_batch, shuffle=(sample_type == 'train')) 
            else:                               #high-level
                subtasks = [Hierarchical_Task(subtask_gen, *batch_dicts_next) for subtask_gen in task_samples]
                return Meta_DataLoader(get_Dataset(param_samples, subtasks), batch_size=mini_batch)

    return {key: get_dataloader(key, mini_batch[key]) for key in ['train', 'test', 'valid']}

#####################
# Task_sampler: generates tasks sampled from task distribution (param_fnc/task_fnc)
class Task_sampler():
    def __init__(self, task_gen, k_batches): 
        task_fnc, param_fnc = task_gen
        if isinstance(task_fnc, dict):
            assert param_fnc is None
            param_fnc = list(task_fnc.keys())   # assume uniform distribution on the list elements
            task_fnc = task_fnc.get
        self.task_fnc  = task_fnc                 # function generates tasks from task-params 
        # self.param_fnc = param_fnc  
        self.params = sample_shuffle_split(param_fnc, k_batches)       # sample k_batches of params 

    def data(self, sample_type): 
        params = self.params[sample_type]
        return params, batch_wrapper(self.task_fnc)(params)    # input / output :  task_params / tasks

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

    k_list = make_list(k_batches)
    total_batch = sum(k_list)

    if input is None:
        input = list(range(total_batch))
    inputs = input(total_batch) if callable(input) else random.sample(input, total_batch)   #  # inputs = sample(input, sum(k_list))     # list of sampled params
    # assert isinstance(inputs, (list, torch.FloatTensor)), "wrong type of inputs"  # , np.ndarray
    
    ###  shuffle and split and make_dict 
    idx = np.cumsum([0]+k_list);   # np.cumsum(filter(None, [0]+k_list))  #  # total_batch = idx[-1]

    return make_dict([inputs[idx[i]:idx[i+1]] for i in range(len(idx)-1)])
