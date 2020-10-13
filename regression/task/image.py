##############################################################################
# Parameter and task sampling functions

import os
import copy
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

img_root = None
train_imgs = []
valid_imgs = []
test_imgs = []
img_size = (32, 32, 3)

def get_img(sample_type):
    img_files = None
    if sample_type == 'train':
        img_files = train_imgs
    elif sample_type == 'valid':
        img_files = valid_imgs
    elif sample_type == 'test':
        img_files = test_imgs
    else:
        raise Exception('Wrong sampling type')

    # Define transforms
    transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                transforms.Resize((img_size[0], img_size[1]), Image.LANCZOS),
                                transforms.ToTensor(),
                                ])
    
    # Randomly choose image, load and return
    img_file = np.random.choice(img_files)
    img_path = os.path.join(img_root, img_file)
    img = transform(img_path).float()
    img = img.permute(1, 2, 0)

    return img

def sample_img_fnc(sample_type):
    img = get_img(sample_type)
    
    def input_function(batch_size, order_pixels=False):
        if order_pixels:
            flattened_indices = list(range(img_size[0] * img_size[1]))[:batch_size]
        else:
            flattened_indices = np.random.choice(list(range(img_size[0] * img_size[1])), batch_size, replace=False)
        
        x, y = np.unravel_index(flattened_indices, (img_size[0], img_size[1]))
        coordinates = np.vstack((x, y)).T
        coordinates = torch.from_numpy(coordinates).float()
        
        # Normalize coordinates
        coordinates[:, 0] /= img_size[0]
        coordinates[:, 1] /= img_size[1]
        return coordinates


    def target_function(coordinates):
        c = copy.deepcopy(coordinates)
        
        # Denormalize coordinates
        c[:, 0] *= img_size[0]
        c[:, 1] *= img_size[1]
        pixel_values = img[c[:, 0].long(), c[:, 1].long(), :]
        return pixel_values

    return input_function, target_function

def load_img_list(data_root, data_split_file):
    global train_imgs, valid_imgs, test_imgs, img_root

    # Root directory for images
    img_root = data_root

    # Saving each split's filenames as list
    with open(data_split_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        train_imgs = []
        valid_imgs = []
        test_imgs = []
        for row in csv_reader:
            if row[1] == '0':
                train_imgs.append(row[0])
            elif row[1] == '1':
                valid_imgs.append(row[0])
            elif row[1] == '2':
                test_imgs.append(row[0])