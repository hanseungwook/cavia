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


# img_root = None
# task = ''
# train_imgs = []
# valid_imgs = []
# test_imgs = []
# train_classes = []
# test_classes = []
# reg_input_range = [-5, 5]

# seed = 2020

### TODO: Can we make k_batch, n_batch automatic?
### TODO: Can we make the sampling of classes mutually exclusive?

dataset_dict = {
    'mnist' : datasets.MNIST,
    'fmnist': datasets.FashionMNIST, 
    'celebA': datasets.CelebA,
    # 'cifar10': datasets.Cifar10, 
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


def img_reconst_task_gen(data_name, root = None): #dataset):
    root = root or os.path.join(os.getcwd(),'data')
    # root    = '/nobackup/users/benhuh/data/'   # shared data directory on satori

    dataset_ = dataset_dict[data_name]
    data_dir = root #os.path.join(root, data_dir)
    
    transforms_ = transforms_dict[data_name]
    img_size = img_size_dict[data_name]

    ############################

    def get_dataset(sample_type):
        return dataset_(data_dir, train=(sample_type == 'train'), transform=transforms_, download=True)    

    def get_labeled_dataset(label, sample_type = 'train'):  # only use images-data from 'train.pth'
        dataset = get_dataset(sample_type)
        targets = dataset.targets if torch.is_tensor(dataset.targets) else torch.tensor(dataset.targets)    # for cifar10
#         if label is not None:   # Huh: if label is None: reduce a level: lv2 -> lv1 (not picking any labels)
        dataset.data = dataset.data[targets == label]
        dataset.target = None
#         insert shuffle
        return dataset

    def get_image(labeled_dataset, idx):
        img, _ = labeled_dataset[idx]
        return img.permute(1, 2, 0)


    def get_pixel(img, xy):
        assert img.shape[2] <= 3, "color dimension should be at the end"       # Usual H x W x C img dimensions

        xy = copy.deepcopy(xy)  # Huh? why?
        x = xy[:, 0] * img_size[0];  
        y = xy[:, 1] * img_size[1]      # restore coordinates
        return img[x.long(), y.long(), :]    
                
    ####################################
    def img_recon_lv2_fnc(label, idx, xy):
        # print(label, idx, xy)
        labeled_dataset = get_labeled_dataset(label)  # level2 
        img = get_image(labeled_dataset, idx)         # level1 
        return get_pixel(img,xy)                      # level0

    def input_fnc_2d_coord(batch, order_pixels=False):
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


    return img_recon_lv2_fnc, input_fnc_2d_coord 



