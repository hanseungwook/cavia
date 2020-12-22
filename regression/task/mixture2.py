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


img_root = None
task = ''
train_imgs = []
valid_imgs = []
test_imgs = []
train_classes = []
test_classes = []
reg_input_range = [-5, 5]
img_size = (32, 32, 3)
seed = 2020

def sample_sin_fnc(sample_type):
    return regression_input_function, get_sin_function(*get_sin_params())

def sample_linear_fnc(sample_type):
    return regression_input_function, get_linear_function(*get_linear_params())

def regression_input_function(batch_size, full=False):
    # Full inputs over the whole regression input range
    if batch_size == 0:
        return torch.linspace(reg_input_range[0], reg_input_range[1], steps=100).unsqueeze(1)

    inputs = torch.randn(batch_size, 1)
    inputs = inputs * (reg_input_range[1] - reg_input_range[0]) + reg_input_range[0]
    return inputs

def get_celeba_img(sample_type):
    global task
    img_files = None

    if not (train_imgs and valid_imgs and test_imgs) or task is not 'celeba':
        load_celeba_img_list('/nobackup/users/swhan/data/Celeba/Img/img_align_celeba', '/nobackup/users/swhan/data/Celeba/Eval/list_eval_partition.txt')
        task = 'celeba'
        
    # Read from global variables
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

def get_cifar10_img(sample_type, label):
    global task
    imgs = None

    if not (train_imgs and test_imgs) or task is not 'cifar10':
        ### TODO: download & re-organize functions
        load_cifar10_imgs('/nobackup/users/swhan/data/cifar-10-batches-py')
        task = 'cifar10'
        
    # Read from global variables
    if sample_type == 'train':
        imgs = train_imgs
    elif sample_type == 'test':
        imgs = test_imgs
    else:
        raise Exception('Wrong sampling type')

    
    # Define transforms
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Randomly choose image given class label, load and return
    imgs = imgs[label]
    img_idx = np.random.randint(low=0, high=imgs.shape[0])
    
    img = imgs[img_idx]
    img = img.transpose(1, 2, 0)
    img = transform(img).float()
    img = img.permute(1, 2, 0)

    return img

def get_mnist_img(sample_type, label):
    global task
    imgs = None

    if not (train_imgs and test_imgs) or task is not 'mnist':
        load_mnist_imgs()
        task = 'mnist'
    
    # Read from global variables
    if sample_type == 'train':
        imgs = train_imgs
    elif sample_type == 'test':
        imgs = test_imgs
    else:
        raise Exception('Wrong sampling type')

    # Get indices of given label in the dataset
    labels = imgs.targets.numpy()
    img_idx = np.random.choice(np.where(labels == label), size=1)

    img, _ = imgs[img_idx]

    return img


def img_input_function(batch_size, order_pixels=False):
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
    if img.shape[2] == 3:
        pixel_values = img[c[:, 0].long(), c[:, 1].long(), :]    
    # Pytorch C x H x W img dimensions
    elif img.shape[0] == 3:
        pixel_values = img[:, c[:, 0].long(), c[:, 1].long()].permute(1, 0) 

    return pixel_values

def setup_cifar10_dataset():
    np.random.seed(seed)
    global train_classes, test_classes
    train_classes = np.random.choice(range(1,10), 7, replace=False)
    test_classes = [l for l in range(1,10) if l not in train_classes]

### TODO: Can we make k_batch, n_batch automatic?
### TODO: Can we make the sampling of classes mutually exclusive?
def sample_cifar10_img_fnc(label, sample_type):
    # if not (len(train_classes) > 0 and len(test_classes) > 0):
    #     setup_cifar10_dataset()
    
    # labels = train_classes if 'train' in sample_type else test_classes
    # label = np.random.choice(labels, 1)[0]
    
    # To make labels non-overlapping, remove label after selection at level 1
    # labels.remove(label)
    img = get_cifar10_img(sample_type, label)
    t_fn = partial(img_target_function, img)

    return img_input_function, t_fn

def sample_mnist_img_fnc(label, sample_type):
    img = get_mnist_img(sample_type, label)
    t_fn = partial(img_target_function, img)

    return img_input_function, t_fn


def sample_celeba_img_fnc(sample_type):
    img = get_celeba_img(sample_type)
    t_fn = partial(img_target_function, img)

    return img_input_function, t_fn

def create_hier_imagenet_supertasks(data_dir, info_dir, level=4, Nsubclasses=20):
    # Setting global parameters for hierarchical imagenet supertask
    global img_size
    # Check if we want these dimensions
    img_size = (256, 256, 3)

    from robustness.tools.breeds_helpers import BreedsDatasetGenerator
    from robustness.tools.breeds_helpers import setup_breeds
    from robustness.tools.breeds_helpers import ClassHierarchy

    # Set up class hierarchy info, if not exist
    if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
        print("Downloading class hierarchy information into `info_dir`")
        setup_breeds(info_dir)

    # Selects all superclasses with at least Nsubclasses # of subclasses
    DG = BreedsDatasetGenerator(info_dir)
    superclasses, _, _ = DG.get_superclasses(level, Nsubclasses, split='rand', ancestor=None, balanced=False, random_seed=2, verbose=False)

    hier = ClassHierarchy(info_dir)

    # Creates supertasks of those superclasses that fit the criterion above
    supertasks = [partial(sample_hier_imagenet_img_fnc, hier, data_dir, info_dir, s, Nsubclasses) for s in superclasses]

    return supertasks


def sample_hier_imagenet_img_fnc(hier, data_dir, info_dir, superclass_id, Nsubclasses, sample_type):
    from robustness.tools.breeds_helpers import BreedsDatasetGenerator
    from robustness.tools.breeds_helpers import ClassHierarchy
    from robustness import datasets

    hier = ClassHierarchy(info_dir)    
    rng = np.random.RandomState(seed)
    
    # Get all subclasses (level 1 classes) and split into train and test (deterministically given a random seed)
    DG = BreedsDatasetGenerator(info_dir)
    total_subclasses = DG.split_superclass(superclass_id, Nsubclasses, True, 'rand', rng)
    
    # Choose between train and valid/test
    subclasses = total_subclasses[0] if sample_type == 'train' else total_subclasses[1]
    subclasses = [[s] for s in subclasses]

    # Define dataset transforms
    # Special transforms for ImageNet(s)
    train_transforms = transforms.Compose([
            transforms.Resize(img_size[0]),
            transforms.CenterCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(img_size[0]),
        transforms.CenterCrop(img_size[0]),
        transforms.ToTensor(),
    ])

    dataset = datasets.CustomImageNet(data_dir, subclasses, transform_train=train_transforms, transform_test=test_transforms)
    loader = dataset.make_loaders(workers=4, batch_size=1)[0]
    img = next(iter(loader))[0].squeeze()
    t_fn = partial(img_target_function, img)

    return img_input_function, t_fn


def load_celeba_img_list(data_root, data_split_file):
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

# TODO: Add download/sort code for CIFAR10 dataset
def load_cifar10_imgs(data_root):
    global train_imgs, test_imgs

    train_imgs = np.load(os.path.join(data_root, 'cifar10_train.npz'), allow_pickle=True)['imgs'].item()
    test_imgs = np.load(os.path.join(data_root, 'cifar10_test.npz'), allow_pickle=True)['imgs'].item()

def load_mnist_imgs():
    global train_imgs, test_imgs, img_size
    img_size = (28, 28, 1)

    train_transforms = transforms.Compose([
            transforms.Resize(img_size[0]),
            transforms.CenterCrop(img_size[0]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(img_size[0]),
        transforms.CenterCrop(img_size[0]),
        transforms.ToTensor(),
    ])

    train_imgs = datasets.MNIST('/nobackup/users/swhan/data/', train=True, transform=train_transforms)
    test_imgs = datasets.MNIST('/nobackup/users/swhan/data/', train=False, transform=test_transforms)

def get_sin_params():
    # Sample n_batch number of parameters
    amplitude = np.random.uniform(0.1, 5.)
    phase = np.random.uniform(0., np.pi)
    return amplitude, phase

def get_sin_function(amplitude, phase):
    def sin_function(x):
        return np.sin(x - phase) * amplitude

    return sin_function

def get_linear_params():
    slope = np.random.uniform(-3., 3.)
    bias = np.random.uniform(-3., 3.)

    return slope, bias

def get_linear_function(slope, bias):
    def linear_function(x):
        return slope * x + bias

    return linear_function


### TODO: visualize images

### TODO: bash script for running script


# def get_quadratic_params():
#     slope1 = np.random.uniform(-0.2, 0.2)
#     slope2 = np.random.uniform(-2.0, 2.0)
#     bias = np.random.uniform(-3., 3.)

#     return slope1, slope2, bias

# def get_quadratic_function(slope1, slope2, bias):
#     def quadratic_function(x):
#         return slope1 * np.square(x, 2) + slope2 * x + bias
#         # TypeError: return arrays must be of ArrayType
#     return quadratic_function

# def get_cubic_params():
#     slope1 = np.random.uniform(-0.1, 0.1)
#     slope2 = np.random.uniform(-0.2, 0.2)
#     slope3 = np.random.uniform(-2.0, 2.0)
#     bias = np.random.uniform(-3., 3.)

#     return slope1, slope2, slope3, bias

# def get_cubic_function(slope1, slope2, slope3, bias):
#     def cubic_function(x):
#         return \
#             slope1 * np.power(x, 3) + \
#             slope2 * np.power(x, 2) + \
#             slope3 * x + \
#             bias

#     return cubic_function

# task_func_list = [
#              (get_sin_params, get_sin_function),
#              (get_linear_params, get_linear_function),
#             #  (get_quadratic_params, get_quadratic_function),
#             #  (get_cubic_params, get_cubic_function),
#             ]
