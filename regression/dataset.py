import random
from torch.utils.data import Dataset #, DataLoader #, Subset
from torch.utils.data.sampler import RandomSampler, BatchSampler

# import IPython
from pdb import set_trace

DOUBLE_precision = False #True
print_task_loader = True
print_loader_type = False
print_hierarhicial_task = False

##############################

def get_Dataset(data, target = None):
    if isinstance(target,list) and isinstance(target[0],dict):
    # if isinstance(target,list) and isinstance(target[0],dict) and data is None:
        return Basic_Dataset_LQR(target)
    else: 
        # print('non-LQR data!')
        return Basic_Dataset(data, target)


##############################
# Basic_Dataset

class Basic_Dataset(Dataset):
    def __init__(self, input, target=None):
        self.input = input
        self.target = target
        if target is not None:
            assert len(input) == len(target)

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        if self.target is None:
            return self.input[idx]
        else: 
            return self.input[idx], self.target[idx], idx

##############################
# Basic_Dataset_LQR

class Basic_Dataset_LQR(Dataset):  # Temporary code. to be properly written
    def __init__(self, target):
        self.kbm = target[0]['kbm']   # same kbm
        self.goal= target[0]['goal']  # same goal
        self.x0 = []
        # names = ['kbm', 'goal', 'x0']
        for t in target:
            self.x0.append(t['x0'])
            # for name in names: 
            #     getattr(self, name).append(t[name])
    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, idx):
        return self.kbm, self.goal, self.x0[idx]
    

##################################
# Meta_DataLoader
class Meta_DataLoader():
    def __init__(self, dataset, batch_size, idx): #, name
        self.dataset = dataset             # pre-sampled list of tasks
        self.minibatch_size = max(1, min(batch_size, len(dataset)))   #   1<=batch_size<=len(dataset)
        self.task_idx = idx
        # self.task_name = name

    def create_mini_batches(self):
        # Create indices of batches
        batch_idx = list(BatchSampler(RandomSampler(self.dataset), self.minibatch_size, drop_last=True))
        
        # since collate does not work for tasks (only works for torch.tensors / arrays ...)
        # Create dataset of minibatches of the form [mini-batch of (hierarchical tasks), ...]
        mini_dataset = []
        for mini_batch_idx in batch_idx:  
            mini_batch = [self.dataset[idx_] for idx_ in mini_batch_idx]  # always same mini-batch.. Todo: change each time? 
            mini_dataset.append(mini_batch)

        return mini_dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        mini_dataset = self.create_mini_batches()
        return iter(mini_dataset)


