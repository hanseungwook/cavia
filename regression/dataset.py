from torch.utils.data import Dataset, DataLoader #, Subset
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
        if target is not None:
            assert len(input) == len(target)

        self.input = input
        self.target = target
        self.named_input = isinstance(self.input[0], (str,int))

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        if self.target is None:
            return self.input[idx]
        else: 
            idx_output = self.input[idx] if self.named_input else idx
            return self.input[idx], self.target[idx], idx_output

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
class Meta_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        batch_size  = max(1, min(batch_size, len(dataset)))          # assert  1<=batch_size<=len(dataset)
        super().__init__(dataset, batch_size)

    def create_mini_batches(self):                      # Create list of minibatch indices
        batch_idx = list(BatchSampler(RandomSampler(self.dataset), self.batch_size, drop_last=True))
        
        # since collate does not work for tasks (only works for torch.tensors / arrays ...)
        # Create dataset of minibatches of the form [mini-batch of (hierarchical tasks), ...]
        mini_dataset = []
        for mini_batch_idx in batch_idx:  
            mini_dataset.append( [self.dataset[idx_] for idx_ in mini_batch_idx] )
        return mini_dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.create_mini_batches())  # create different set of mini-batch each time __iter__ is called. 
         

########################
# testing code 

def test_dataloader():
    dataset = Basic_Dataset(list(range(6)), list(range(6)))
    dataloader = Meta_DataLoader(dataset, batch_size=2)  # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i in range(4):
        print('new iter!')
        for data in dataloader:
            print(data)