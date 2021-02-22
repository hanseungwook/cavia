# lv3: dataset f( .  , rnd, type)    -> dataset    (e.g. MNIST vs fMNIST)
# lv2: label   f(dataset, rnd, type) -> label                -> Loss( theta | dataset )           -> averaged over labels
# lv1: image   f(label, rnd, type)   -> image                -> Loss( ctx1 | label | theta)       -> averaged over image
# lv0: pixel   f(image, rnd, type)   -> xy, pixel (target)   -> Loss( ctx0 | image | ctx1, theta) -> averaged over xy E_xy[L(xy)]

##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]

import random
from torch.utils.data import Dataset, DataLoader #, Subset
from torch.utils.data.sampler import RandomSampler, BatchSampler

# import IPython
from pdb import set_trace

DOUBLE_precision = False #True
print_task_loader = True
print_loader_type = False
print_hierarhicial_task = False


class Basic_Dataset(Dataset):
    def __init__(self, data, target=None):
        self.data = data
        self.target = target
        if target is not None:
            assert len(data) == len(target)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.target is None:
            return self.data[idx]
        else: 
            return self.data[idx], self.target[idx]

class Basic_Dataset_LQR(Dataset):  # Temporary code. to be properly written
    def __init__(self, data, target=None):
        assert isinstance(data,tuple) and len(data)==3
        kbm, goal, x0 = data
        self.kbm = kbm
        self.goal = goal
        self.x0 = x0

    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, idx):
        return self.kbm, self.goal, self.x0[idx]
    
def Dataset_helper(data, target = None):
    if isinstance(data,tuple):
        assert len(data)==3
        return Basic_Dataset_LQR(data)
    else: 
        return Basic_Dataset(data, target)

##################################

class Meta_DataLoader():
    def __init__(self, dataset, batch_size, name, idx):
        self.dataset = dataset               # pre-sampled list of tasks
        self.minibatch_size = batch_size
        self.task_name = name
        self.task_idx = idx

        assert self.minibatch_size <= len(dataset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # Create indices of batches
        batch_idx = list(BatchSampler(RandomSampler(self.dataset), self.minibatch_size, drop_last=True))

        # since collate does not work for tasks (only works for torch.tensors / arrays ...)
        # Create dataset of minibatches of the form [mini-batch of (hierarchical tasks), ...]
        mini_dataset = []
        for mini_batch_idx in batch_idx:  
            mini_batch = [self.dataset[idx_] for idx_ in mini_batch_idx]  # always same mini-batch.. Todo: change each time? 
            mini_dataset.append(mini_batch)

        return iter(mini_dataset)
    



##############################################################################


class Hierarchical_Task():   
    # Top-down generation of task hierarchy.
    def __init__(self, task, batch_dict, idx=0): 
        level = len(batch_dict[0]) - 1          # print(self.level, total_batch_dict)
        self.loader_dict = get_dataloader_dict(level, task, batch_dict, idx)
        
#         if print_hierarhicial_task:
#             print('level', len(batch_dict[0]) - 1)
#             print('batch', batch_dict[0][-1]['train'])
#             print('task', task)
        
#         if print_task_loader and level>0:
#             print('Task_loader Level', level, 'task', task)
                    
    def load(self, sample_type):   
        if sample_type == 'train':
            return self.loader_dict[sample_type]     # return dataloader
        else: 
            return next(iter(self.loader_dict[sample_type]))   # return one iter from dataloader

def get_dataloader_dict(level, task, batch_dict, idx):    
    batch_dict_next = (batch_dict[0][:-1], batch_dict[1][:-1])
    total_batch, mini_batch = batch_dict[0][-1], batch_dict[1][-1]           # mini_batch: mini batch # of samples
    task.pre_sample(total_batch)

    def get_dataloader(sample_type, mini_batch_):     #     sample_type = 'train' or 'test'  
        if level == 0:
            input_data, target = task.get_data(sample_type) #, total_batch_)
            return DataLoader(Dataset_helper(input_data, target), batch_size=mini_batch_, shuffle=(sample_type == 'train'))  # returns tensors
        
        else:
            task_params, subtask_list = task.get_data(sample_type) #, total_batch_) 
            data = [Hierarchical_Task(subtask, batch_dict_next, idx_) for (idx_, subtask) in enumerate(subtask_list)]   # Recursive
            dataloader = Meta_DataLoader(Dataset_helper(data, None), batch_size=mini_batch_, name=str(task), idx=idx)  # returns a minibatch of Tasks
#             print(sample_type, task_params) 
            return dataloader

    loader_dict = {key: get_dataloader(key, mini_batch[key]) for key in ['train', 'test', 'valid']}
    return loader_dict

