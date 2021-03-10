import torch
import numpy as np
import random

from dataset import get_Dataset
from torch.utils.data import DataLoader 
from torch.utils.data._utils.collate import default_convert, default_collate_err_msg_format
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

shuffle = True  #False # # None

class Hierarchical_Task():   
    # Top-down generation of task hierarchy.
    
    def __init__(self, task_gen, total_batches, mini_batches):
        total_batch, mini_batch = total_batches[-1], mini_batches[-1]     # for current level  
        batch_dicts_next = (total_batches[:-1], mini_batches[:-1])        # for next level 

        sampler = Task_sampler(task_gen, total_batch, shuffle) 
        self.dataloaders = get_dataloader_dict(sampler, mini_batch, batch_dicts_next, shuffle) 

    def load(self, sample_type):   
        return self.dataloaders[sample_type] or self.dataloaders['train']  # duplicate 'train' loader, if 'test'/'valid' loader == None
        
#####################
# get_dataloader


def get_dataloader_dict(sampler, mini_batch, batch_dicts_next, shuffle_ = None):

    def get_dataloader(sample_type, mini_batch):   
        param_samples, task_samples = sampler.data(sample_type)

        shuffle =(sample_type == 'train') if shuffle_ is None else shuffle_

        if len(param_samples) == 0:     # No dataloader for empty data samples
            return None
        else:
            if len(batch_dicts_next[0]) == 0:    # level0:
                inputs, targets = param_samples, task_samples  # re-name into inputs/targets
                return continuous_loader(DataLoader(get_Dataset(inputs, targets), batch_size=mini_batch, shuffle = shuffle) )
            else:                               #high-level
                subtasks = [Hierarchical_Task(subtask_gen, *batch_dicts_next) for subtask_gen in task_samples]
                return continuous_loader(DataLoader(get_Dataset(param_samples, subtasks), batch_size=mini_batch, collate_fn=custom_collate, shuffle = shuffle ))

    return {key: get_dataloader(key, mini_batch[key]) for key in ['train', 'test', 'valid']}

#####################
# Task_sampler: generates tasks sampled from task distribution (param_fnc/task_fnc)
class Task_sampler():
    def __init__(self, task_gen, k_batches, shuffle): 
        task_fnc, param_fnc = task_gen
        if isinstance(task_fnc, dict):
            assert param_fnc is None
            param_fnc = list(task_fnc.keys())   # assume uniform distribution on the list elements
            task_fnc = task_fnc.get
        self.task_fnc  = task_fnc                 # function generates tasks from task-params 
        # self.param_fnc = param_fnc  
        self.params = sample_shuffle_split(param_fnc, k_batches, shuffle)       # sample k_batches of params 

    def data(self, sample_type): 
        params = self.params[sample_type]
        return params, batch_wrapper(self.task_fnc)(params)    # input / output :  task_params / tasks

########################
# sample_shuffle_split

def sample_shuffle_split(input, k_batches: dict, shuffle):
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
    # inputs = input(total_batch) if callable(input) else random.sample(input, total_batch)   #  # inputs = sample(input, sum(k_list))     # list of sampled params

    if callable(input) :
        inputs = input(total_batch) 
    else:
        inputs = random.sample(input, total_batch) if shuffle else input[:total_batch] 
    # assert isinstance(inputs, (list, torch.Tensor)), "wrong type of inputs"  # , np.ndarray
    
    ###  shuffle and split and make_dict 
    idx = np.cumsum([0]+k_list);   # np.cumsum(filter(None, [0]+k_list))  #  # total_batch = idx[-1]

    return make_dict([inputs[idx[i]:idx[i+1]] for i in range(len(idx)-1)])


###############################
# collate function
import collections
from torch._six import string_classes


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # set_trace()
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, Hierarchical_Task):
        return batch
    elif isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        set_trace()
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


########################
#  continuous_loader    
class continuous_loader():  # wrapper
    def __init__(self, loader):
        self.loader = loader
        self.iterator = self.get_iter()

    def get_iter(self):
        # if isinstance(self.loader, list):
        #     return [iter(l) for l in self.loader]
        # else:
            return iter(self.loader)

    def get_next(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = self.get_iter()
            data = next(self.iterator)
        return data
