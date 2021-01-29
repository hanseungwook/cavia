import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torchvision.transforms import transforms
import copy
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import IPython

from pdb import set_trace

# from task.mixture2 import task_func_list


# class Level0_Dataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
    
#     def __len__(self):
#         assert len(self.x) == len(self.y)

#         return len(self.y)
    
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

class Meta_Dataset(Dataset):
    def __init__(self, data, target=None):
        if isinstance(data,tuple) and len(data)==3:
            kbm, target, x0 = data
            self.LQR_flag = True
            self.kbm = kbm
            self.target = target
            self.data = x0
        else:
            self.LQR_flag = False
            self.data = data
            self.target = target
            if target is not None:
                assert len(data) == len(target)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
#         if self.__len__()>1:
#             set_trace()
        if self.LQR_flag:
            return self.kbm, self.target, self.data[idx]
        
        else:
            if self.target is None:
                return self.data[idx]
            else: 
                return self.data[idx], self.target[idx]


class Meta_DataLoader():
    def __init__(self, dataset, batch_size, task_name):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_name = task_name

    def __iter__(self):
        # Create indices of batches
        batch_idx = list(BatchSampler(RandomSampler(self.dataset), self.batch_size, drop_last=True))

        # Create dataset of minibatches of the form [mini-batch of (hierarchical tasks), ...]
        mini_dataset = []
        for mini_batch_idx in batch_idx:
            mini_batch = [self.dataset[idx] for idx in mini_batch_idx]
            mini_dataset.append(mini_batch)

        return iter(mini_dataset)

