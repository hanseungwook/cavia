# lv3: dataset f( .  , rnd, type)    -> dataset    (e.g. MNIST vs fMNIST)
# lv2: label   f(dataset, rnd, type) -> label                -> Loss( theta | dataset )           -> averaged over labels
# lv1: image   f(label, rnd, type)   -> image                -> Loss( ctx1 | label | theta)       -> averaged over image
# lv0: pixel   f(image, rnd, type)   -> xy, pixel (target)   -> Loss( ctx0 | image | ctx1, theta) -> averaged over xy E_xy[L(xy)]


import random
from torch.utils.data import Dataset, DataLoader #, Subset
from torch.utils.data.sampler import RandomSampler, BatchSampler

from task.Siren_dataio import get_mgrid

import IPython
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
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]


# class Hierarchical_Task():  
#     def __init__(self, task, batch_dict, idx=0): #task, batch_dict, idx): 
#         self.idx = idx
#         level = len(batch_dict[0]) - 1          # print(self.level, total_batch_dict)
        
#         if print_hierarhicial_task:
#             print('level', len(batch_dict[0]) - 1)
#             print('batch', batch_dict[0][-1]['train'])
#         print('task', task)
        
#         self.loader_dict = get_dataloader_dict(level, task, batch_dict, idx)
        
#         if print_task_loader and level>0:
#             print('Task_loader Level', level, 'task', task)
            
#     def load(self, sample_type):   
#         if sample_type == 'train':
#             return self.loader_dict[sample_type]     # return dataloader
#         else: 
#             return next(iter(self.loader_dict[sample_type]))   # return one iter from dataloader

# ###########################

            
# def get_dataloader_dict(level, task, batch_dict, idx):    
    
#     def get_dict():
#         total_batch_dict, mini_batch_dict = batch_dict
#         batch_dict_next = (total_batch_dict[:-1], mini_batch_dict[:-1])
#         total_batch = total_batch_dict[-1]         # total_batch: total # of samples   
#         mini_batch = mini_batch_dict[-1]           # mini_batch: mini batch # of samples

#         loader_dict = {}
#         for sample_type in  ['train', 'test', 'valid']:
#             if print_loader_type:
#                 print('level', level, 'type', sample_type)
#             loader_dict[sample_type] = get_dataloader(sample_type, total_batch[sample_type], mini_batch[sample_type], batch_dict_next)
#         return loader_dict
    
#     def get_dataloader(sample_type, total_batch, mini_batch, batch_dict_next):     #     sample_type = 'train' or 'test'  

#         def sample_base_data(task):             # make a dataset out of the samples from the given task
#             assert isinstance(task,tuple)
#             input_generator, target_generator = task  # Generator functions for the input and target
#             input_data  = input_generator(total_batch, sample_type)  # Huh : added sample_type as input
#             target = target_generator(input_data) if target_generator is not None else None
#         #     if DOUBLE_precision:
#         #         input_data  = input_data.double();  target = target.double() if target is not None else None  
#             return input_data, target
        
#         def sample_meta_data(task):             # make a dataset out of the samples from the given task
# #             print(task)
# #             set_trace()
# #             assert hasattr(task, 'get_data')
# #             assert callable(task)
            
# #             if isinstance(task, list):       
# #                 assert total_batch <= len(task)
# #                 subtask_list = random.sample(task, total_batch)    #  sampling from task list # To fix: task does not take sample_type as an input
# #             elif callable(task): 
# #                 subtask_list = task(sample_type) #[task(sample_type) for _ in range(total_batch)]  #  sampling from task_generating function 
# #             else: 
# #                 print(task)
# #                 error()
            
#             params, subtask_list = task.get_data(sample_type) 
#             data = [Hierarchical_Task(subtask, batch_dict_next, idx_) for (idx_, subtask) in enumerate(subtask_list)]   # Recursive
#             return data


#         set_trace()
#         print('level', level, 'task', task)
#         if level == 0:
#             data, target = sample_base_data(task) 
#             dataset = Dataset_helper(data, target)
#             if mini_batch == 0 and total_batch == 0:     # To fix:  why assume total_batch == 0   ??
#                 mini_batch = len(dataset)                  # Full range/batch if both 0s
#             return DataLoader(dataset, batch_size=mini_batch, shuffle=(sample_type == 'train'))                # returns tensors
        
#         else:
#             subtask_samples = sample_meta_data(task)      # list of sampled subtasks
#             subtask_dataset = Dataset_helper(subtask_samples, None)
#             return Meta_DataLoader(subtask_dataset, batch_size=mini_batch, name=str(task), idx=idx)      #  returns a mini-batch of Hiearchical Tasks[

#     return get_dict()



class Hierarchical_Task():   
    # Top-down generation of task hierarchy.
    def __init__(self, task, batch_dict, idx=0): #task, batch_dict, idx): 
        level = len(batch_dict[0]) - 1          # print(self.level, total_batch_dict)
        self.loader_dict = get_dataloader_dict(level, task, batch_dict, idx)
        
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
            # print(sample_type, task_params) #data, idx, task)
            return dataloader

    loader_dict = {key: get_dataloader(key, mini_batch[key]) for key in ['train', 'test', 'valid']}
    return loader_dict


###############################

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



#################################



class Implicit2DWrapper(Dataset):
    def __init__(self, dataset, sidelength): #, img2fnc=None):

#         if isinstance(sidelength, int):
#             sidelength = (sidelength, sidelength)
#         self.sidelength = sidelength

        self.transform = Compose([ Resize(sidelength), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])) ])
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)
#         self.img2fnc = img2fnc

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': self.img2fnc(img)}

        return in_dict, gt_dict
    
def img2fnc(mgrid, img):
    subsamples = int(self.test_sparsity)
    rand_idcs = np.random.choice(img.shape[0], size=subsamples, replace=False)
    img_sparse = img[rand_idcs, :]
    coords_sub = self.mgrid[rand_idcs, :]
    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub}

#     def get_item_small(self, idx):   ### used in dataio.ImageGeneralizationWrapper()
#         img = self.transform(self.dataset[idx])
#         spatial_img = img.clone()
#         img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

#         gt_dict = {'img': img}

#         return spatial_img, img, gt_dict
    
    
################


# class CelebA(Dataset):
#     def __init__(self, split, downsampled=False):
#         # SIZE (178 x 218)
#         super().__init__()
#         assert split in ['train', 'test', 'val'], "Unknown split"

#         self.root = '/media/data3/awb/CelebA/kaggle/img_align_celeba/img_align_celeba'
#         csv_path = '/media/data3/awb/CelebA/kaggle/list_eval_partition.csv'

#         self.img_channels = 3
#         self.fnames = []

#         with open(csv_path, newline='') as csvfile:
#             rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#             for row in rowreader:
#                 if split == 'train' and row[1] == '0':
#                     self.fnames.append(row[0])
#                 elif split == 'val' and row[1] == '1':
#                     self.fnames.append(row[0])
#                 elif split == 'test' and row[1] == '2':
#                     self.fnames.append(row[0])

#         self.downsampled = downsampled

#     def __len__(self):
#         return len(self.fnames)

#     def __getitem__(self, idx):
#         path = os.path.join(self.root, self.fnames[idx])
#         img = Image.open(path)
#         if self.downsampled:
#             width, height = img.size  # Get dimensions

#             s = min(width, height)
#             left = (width - s) / 2
#             top = (height - s) / 2
#             right = (width + s) / 2
#             bottom = (height + s) / 2
#             img = img.crop((left, top, right, bottom))
#             img = img.resize((32, 32))

#         return img
