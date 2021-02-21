##############################################################################
# Parameter and task sampling functions

import os
from functools import partial 
import copy
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

import IPython
from pdb import set_trace

data_dir    = '/nobackup/users/benhuh/data/'   # shared data directory on satori
celeba_dir  = data_dir + 'Celeba'
cifar10_dir = data_dir + 'cifar-10-batches-py'

img_root = None
task = ''
train_imgs = []
valid_imgs = []
test_imgs = []
train_classes = []
test_classes = []
reg_input_range = [-5, 5]

seed = 2020

### TODO: Can we make k_batch, n_batch automatic?
### TODO: Can we make the sampling of classes mutually exclusive?



    
####################

def img_input_function(img_size, batch_size, order_pixels=False):
    if order_pixels:
        flattened_indices = list(range(img_size[0] * img_size[1]))[:batch_size]
    else:
        # Returning full range (in sorted order) if batch size is the full image size
        if batch_size == 0:
            flattened_indices = list(range(img_size[0] * img_size[1]))
        else: 
            flattened_indices = np.random.choice(list(range(img_size[0] * img_size[1])), batch_size, replace=False)

    x, y = np.unravel_index(flattened_indices, (img_size[0], img_size[1]))
    coordinates = np.vstack((x, y)).T
    coordinates = torch.from_numpy(coordinates).float()
    
    # Normalize coordinates
    coordinates[:, 0] /= img_size[0]
    coordinates[:, 1] /= img_size[1]
    return coordinates

def img_target_function(img, coordinates):
    c = copy.deepcopy(coordinates)
    
    # Denormalize coordinates
    c[:, 0] *= img_size[0]
    c[:, 1] *= img_size[1]

    # Usual H x W x C img dimensions
    if img.shape[2] == 3 or img.shape[2] == 1:
        pixel_values = img[c[:, 0].long(), c[:, 1].long(), :]    
    # Pytorch C x H x W img dimensions
    elif img.shape[0] == 3 or img.shape[0] == 1:
        pixel_values = img[:, c[:, 0].long(), c[:, 1].long()].permute(1, 0) 

    return pixel_values



########################
## For MNIST / FMNIST / CelebA


def get_dataset_1_label(dataset_, sample_type, label):
    dataset_ = dataset(data_dir, train=(sample_type == 'train'), transform=transforms.ToTensor(), download=True)        
    targets = dataset_.targets if dataset_.targets else torch.tensor(dataset_.targets)    # for cifar10
    idx = targets ==label
    dataset_.data = dataset_.data[idx];    dataset_.targets= None #dataset_.targets[idx]
    return dataset_
    

# def get_dataset_multi_label(dataset, sample_type, labels, batch = None):
#     assert isinstance(labels,list)
# #     if not isinstance(labels,list):
# #         labels = [labels]
#     dataset_ = dataset(data_dir, train=(sample_type == 'train'), transform=transforms.ToTensor(), download=True)        
#     targets = dataset_.targets if dataset_.targets else torch.tensor(dataset_.targets)    # for cifar10
        
#     dataset_dict = {}
#     for label in labels:
#         temp = dataset_
#         idx = temp.targets==label
#         set_trace()
#         temp.data = temp.data[idx]
#         temp.targets= None #temp.targets[idx]
            
#         if batch is not None:
#             temp.data = temp.data[idx][:batch] #,:,:]
# #             temp.targets = None #temp.targets[idx][:batch]
#         dataset_dict[label] = temp
#     return dataset_dict

dataset_dict = {
    'mnist': datasets.MNIST,
    'fmnist': datasets.FashionMNIST,
    'celebA': datasets.CelebA,
}

img_size_dict = {
    'mnist': (28, 28, 1),
    'fmnist': (28, 28, 1),
    'celebA': (32, 32, 3),
}

transforms_dict = {
    'mnist': transforms.ToTensor(),
    'fmnist': transforms.ToTensor(),
    'celebA': transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                transforms.Resize(img_size_dict['celebA'][0:2], Image.LANCZOS),  #transforms.Resize((img_size[0], img_size[1]), Image.LANCZOS),
                                transforms.ToTensor() ])
}


def get_image_fnc_helper(data_name): #dataset):

    dataset = dataset_dict[data_name]
    transform = transforms_dict[data_name]
    
#     data_dict = {} 

    def get_labeled(label):
        
# #         for sample_type in ['train', 'test', 'valid']:
# #             data_dict[sample_type] = dataset(data_dir, train=(sample_type == 'train'), transform=transforms.ToTensor(), download=True)        
# #             dataset_[sample_type] = get_dataset_1_label(dataset, sample_type, label)
#             dataset_partial = partial(get_dataset_1_label, dataset, label)
    
        def get_image_fnc(sample_type):
            dataset_ = dataset(data_dir, train=(sample_type == 'train'), transform=transforms.ToTensor(), download=True)        

    #         labels = dataset_.targets.numpy()
    #         dataset__ = get_dataset_1_label(dataset_[sample_type], label)
#             dataset_ = data_dict[sample_type]
            # dataset_ = dataset_partial(sample_type)
        
            labels = dataset_.targets if torch.is_tensor(dataset_.targets) else torch.tensor(dataset_.targets)    # for cifar10

            img_idx = np.random.choice(np.where(labels == label)[0], size=1)[0]
            img, _ = dataset_[img_idx]  # choose a random image
            img = img.permute(1, 2, 0)  # 1x28x28 -> 28x28x1 #
            return img

        return get_image_fnc
    return get_labeled

##########
# idx = dataset.train_labels==1
# dataset.train_labels = dataset.train_labels[idx]
# dataset.train_data = dataset.train_data[idx]


###################################

def sample_fnc_helper(task_name):
    get_img_fnc = get_image_fnc_helper(task_name) #  get_image_fnc_helper(dataset_dict[task_name], transforms_dict[task_name])
    img_size = img_size_dict[task_name]
    
    def sample_fnc_h(label):
        def sample_fnc(sample_type):
            img = get_img_fnc(sample_type, label)
            input_fnc  = partial(img_input_function, img_size)
            target_fnc = partial(img_target_function, img)
            return  input_fnc, target_fnc
        return sample_fnc
    return sample_fnc_h

sample_mnist_img_fnc  = sample_fnc_helper('mnist')
sample_fmnist_img_fnc = sample_fnc_helper('fmnist')


