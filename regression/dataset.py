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


class Level0_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        assert len(self.x) == len(self.y)

        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class HighLevel_Dataset(Dataset):
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

        # Create dataset of minibatches of the form [mini-batch of (hierarchical tasks), ...]
        mini_dataset = []
        for mini_batch_idx in batch_idx:
            mini_batch = [self.dataset[idx] for idx in mini_batch_idx]
            mini_dataset.append(mini_batch)

        return iter(mini_dataset)