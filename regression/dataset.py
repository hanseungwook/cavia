from torch.utils.data import Dataset, DataLoader #, Subset
from torch.utils.data.sampler import RandomSampler, BatchSampler


import IPython

from pdb import set_trace




class Meta_Dataset(Dataset):
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

class Meta_Dataset_LQR(Dataset):  # Temporary code. to be properly written
    def __init__(self, data, target=None):
        assert isinstance(data,tuple) and len(data)==3
        kbm, goal, x0 = data
#         self.LQR_flag = True
        self.kbm = kbm
        self.goal = goal
        self.x0 = x0

    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, idx):
        return self.kbm, self.goal, self.x0[idx]
    
    

##################################

class Meta_DataLoader():
    def __init__(self, dataset, batch_size, task_name, task_idx):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_name = task_name
        self.task_idx = task_idx

    def __iter__(self):
        # Create indices of batches
        batch_idx = list(BatchSampler(RandomSampler(self.dataset), self.batch_size, drop_last=True))

        # Create dataset of minibatches of the form [mini-batch of (hierarchical tasks), ...]
        mini_dataset = []
        for mini_batch_idx in batch_idx:  
            mini_batch = [self.dataset[idx] for idx in mini_batch_idx]  # always same mini-batch.. Todo: change each time? 
            mini_dataset.append(mini_batch)

        return iter(mini_dataset)

# def get_samples(task, total_batch, sample_type):
#     if isinstance(task, dict):
#         # Separate level-2 train and test tasks
#         if task[sample_type]:
#             task = task[sample_type]
#         # Same level-2 train and test tasks 
#         else: 
#             task = task['train']

#         # # For 3-level or above training, tasks will continuously be dicts, not lists
#         # if instance(task, dict):
#         #     # Skip random sampling here for now
#         #     tasks = task
#         # else:
#         assert total_batch <= len(task)
#         tasks = random.sample(task, total_batch)
#     # Levels below 2
#     else:
#         tasks = list(task(sample_type) for _ in range(total_batch))
#     return tasks
