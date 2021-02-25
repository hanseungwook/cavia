import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import copy
# from functools import partial 
# import csv

import IPython
from pdb import set_trace

from hierarchical_task import batch_wrapper

### TODO: Can we make k_batch, n_batch automatic?
### TODO: Can we make the sampling of classes mutually exclusive?

dataset_dict = {
    'mnist' : datasets.MNIST,
    'fmnist': datasets.FashionMNIST, 
    'celebA': datasets.CelebA,
    'cifar10': datasets.Cifar10, 
}

img_size_dict = {
    'mnist':  (28, 28, 1),
    'fmnist': (28, 28, 1),
    'celebA': (32, 32, 3),
    'cifar10': (32, 32, 3), 
}

transforms_dict = {
    'mnist':  transforms.ToTensor(),
    'fmnist': transforms.ToTensor(),
    'cifar10': transform = transforms.Compose( [transforms.ToTensor(),     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'celebA': transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                  transforms.Resize(img_size_dict['celebA'][0:2], Image.LANCZOS), 
                                  transforms.ToTensor() ])
}


def img_reconst_task_gen(data_name, root = None): #dataset):

    root = root or os.path.join(os.getcwd(),'data')
   
    dataset_    = dataset_dict   [data_name]
    transforms_ = transforms_dict[data_name]
    img_size    = img_size_dict  [data_name]
    ############################

    def get_dataset(split = 'train'):
        return dataset_(root, train=(split == 'train'), transform=transforms_, download=True)    

    def get_labeled_dataset(label):  # level0  # default: only use images-data from 'train.pth'
        dataset = get_dataset()
        print(dataset.data.shape)
        targets = dataset.targets if torch.is_tensor(dataset.targets) else torch.tensor(dataset.targets)    # for cifar10
    
        if label is  None:  
            pass # Huh: if label is None: reduce a level: lv2 -> lv1 (not picking any labels)
        else:
            dataset.data = dataset.data[targets == label]
            dataset.target = None
#         insert dataset shuffle
        return dataset

    def get_image(labeled_dataset, idx):  # level1 
        img, _ = labeled_dataset[idx]
        return img.permute(1, 2, 0)


    def get_pixel(img, xy):               # level0
        assert img.shape[2] <= 3, "color dimension should be at the end"       # Usual H x W x C img dimensions

        xy = copy.deepcopy(xy)          # Huh: why deepcopy?
        x = xy[:, 0] * img_size[0];     # restore coordinates
        y = xy[:, 1] * img_size[1]      
        return img[x.long(), y.long(), :]    

    ####################################
    # task_fnc

    # dataset, img_size = get_dataset(data_name)

    # input_fnc
    def xy_coord0_params(batch, order_pixels=False):
        # Returning full range (in sorted order) if batch size is the full image size
        idx_full = list(range(img_size[0] * img_size[1]))
        if batch == 0:
            flattened_indices = idx_full
        else: 
            if order_pixels:
                flattened_indices = idx_full[:batch]
            else:
                flattened_indices = np.random.choice(idx_full, batch, replace=False)

        x, y = np.unravel_index(flattened_indices, (img_size[0], img_size[1]))
        coordinates = np.vstack((x, y)).T
        coordinates = torch.from_numpy(coordinates).float()
        
        # Normalize coordinates
        coordinates[:, 0] /= img_size[0]
        coordinates[:, 1] /= img_size[1]
        return coordinates

    # task_fnc 
    def lv2_fnc(label):
        labeled_dataset = get_labeled_dataset(label)
        def lv1_fnc(idx):
            img = get_image(labeled_dataset, idx)         # level1 
            def lv0_fnc(xy):
                return get_pixel(img,xy)                      # level0
            return lv0_fnc, xy_coord0_params
        return lv1_fnc, None

    return lv2_fnc, None