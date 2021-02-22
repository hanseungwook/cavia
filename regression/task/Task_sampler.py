import torch
import numpy as np
import random
from functools import partial


# from pdb import set_trace



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

    def pre_sample(self, k_batches: dict):              # pre-sample k_batches of params  
        self.params = sample_shuffle_split(self.param_fnc, k_batches)
        ### maybe comment out?
        for type_ in ['test', 'valid']:
            if len(self.params[type_]) == 0:
                self.params[type_] = self.params['train']   # duplicate train and test params.

    def get_data(self, sample_type): 
        if self.params is None: 
            error()
        params = self.params[sample_type]
        return params, batch_wrapper(self.task_fnc)(params) 

#     def __str__(self):
#         return self.param_fnc.__name__ if hasattr(self.param_fnc, '__name__')  else 'None' # self.task_fnc.__name__ #self.params

####
def Partial_Task_sampler(task_fnc, input_fnc): #, which = 'task' ):
    def fnc_(*params):
        return Task_sampler(task_fnc = partial(task_fnc, *params), param_fnc = input_fnc)
    return fnc_

####
def sample_shuffle_split(input, k_batches: dict):

    keys = ['train', 'test', 'valid']
    def make_dict(vals):
        return {k:v for (k,v) in zip(keys, vals)}

    if k_batches is None and hasattr(input, '__len__'): 
        k_batches_list = [len(input),0,0]
    else:
        assert  keys == list(k_batches.keys()), "wrong keys for k_batches dict"
        k_batches_list = list(k_batches.values())
    assert len(k_batches_list) == 3, "wrong number of k_batches_list items"
    total_batch = sum(k_batches_list)
    
    # list of sampled params
    if input is None:
        input_ = list(range(total_batch))
    else:
        input_ = input(total_batch) if callable(input) else input   
        
    ###  shuffle and split and make_dict 
    idx = np.cumsum([0]+k_batches_list);    # total_batch = idx[-1]
    
    assert isinstance(input_, (list, torch.FloatTensor)), "wrong type of input_"  # , np.ndarray
    if isinstance(input_, list): 
        random.shuffle(input_)
        input_dict = make_dict([input_[idx[i]:idx[i+1]] for i in range(len(idx)-1)])
    elif isinstance(input_, torch.FloatTensor):
        rnd_idx = torch.randperm(input_.shape[0])
        input_ = input_[rnd_idx].view(input_.shape)  # shuffled tensor
        input_dict = make_dict([input_[idx[i]:idx[i+1], :] for i in range(len(idx)-1)])

    return input_dict


#####

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
