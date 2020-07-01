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

from task.mixture2 import task_func_list


class CustomList:
    def __init__(self, data, batch_size):
        self._data = data
        self._data_copy = copy.deepcopy(data)
        self.batch_size = batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch_size = min(self.batch_size, len(self._data_copy))

        sample_idx = random.sample(range(len(self._data_copy)), batch_size)
        sample = self._data_copy[sample_idx]

        del self._data_copy[sample_idx]
        return sample
    
class Dataset_XY(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Level0_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        assert len(self.x) == len(self.y)

        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LevelOthers_Dataset(Dataset):
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]


class HighLevel_DataLoader():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # Create indices of batches
        batch_idx = list(BatchSampler(RandomSampler(self.dataset), self.batch_size, drop_last=False))

        # Create dataset of minibatches of the form [Subset(hierarchical tasks), ...]
        mini_dataset = [Subset(self.dataset, idx) for idx in batch_idx]

        return iter(mini_dataset)

# class Hierarchical_Dataset():
#     def __init__(self, batch_dict):
#         self.k_batch_dict, self.n_batch_dict = batch_dict

#         self.datasets = {'train': [Actual instantiation of a function (sinusoid) with parameters], 'test': []}
#         self.pre_sample()
    
#     def pre_sample(self):
#         # Create all lowest level datasets here, so that we can all pre-sample them in the highest level?
#         # Pre-sample train
#         for i in range(len(self.k_batch_dict)):
#             cur_k_batch = self.k_batch_dict[i]
            
#             for j in range(cur_k_batch['train']):

#         # Pre-sample test

#     def sample(self, sample_type, level):
#         # Return lowest level dataset at level 1, otherwise keep returning this hierarchical dataset
#         if level == 1:
#             # Should we return only 1 instance from self.datasets?
#             return self.datasets[sample_type]
#         else:
#             return self


# class Function_Dataset(Dataset):
#     def __init__(self, task_idx):
#         self.task = task_func_list[task_idx]
#         self.pre_sample()

#     def __len__(self):
#         return 10
    
#     def __getitem__(self, index):
#         return self.x[index]
    
#     def pre_sample(self):
        
        

def main():
    # t = [1,2,3,4,5]
    dataset = Level0_Dataset(x=torch.arange(10), y=torch.arange(10))
    t = HighLevel_DataLoader(dataset, 3)
    t_iter = iter(t)

    for cur in t_iter:
        print(list(cur))

if __name__ == "__main__":
    main()