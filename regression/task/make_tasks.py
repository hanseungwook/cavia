from functools import partial
from .regression_1d import sample_sin_fnc, sample_linear_fnc, sample_quadratic_fnc, sample_cubic_fnc
from .image_reconstruction import sample_celeba_img_fnc, sample_cifar10_img_fnc, sample_mnist, sample_mnist_img_fnc, sample_fashion_mnist, sample_fashion_mnist_img_fnc, create_hier_imagenet_supertasks
from .LQR import LQR_environment, Linear_Policy, Combine_NN_ENV


def make_tasks(task_names, classes):
    task_func_dict = {}
    task_func_dict['train'] = []
    task_func_dict['test'] = []

    # If 'test' is not set, 'test' uses the same tasks/labels as 'train'
    for task in task_names:
        if task == 'sine':
            task_func_dict['train'].append(sample_sin_fnc)
        elif task == 'linear':
            task_func_dict['train'].append(sample_linear_fnc)
        elif task == 'quadratic':
            task_func_dict['train'].append(sample_quadratic_fnc)
        elif task == 'cubic':
            task_func_dict['train'].append(sample_cubic_fnc)
        elif task == 'celeba':
            task_func_dict['train'].append(sample_celeba_img_fnc)
        elif task == 'cifar10':
            task_func_dict['train'].extend([partial(sample_cifar10_img_fnc, l) for l in range(7)])
            task_func_dict['test'].extend([partial(sample_cifar10_img_fnc, l) for l in range(7,10)])
        elif task == 'celeba_airplane':
            task_func_dict['train'].extend([partial(sample_cifar10_img_fnc, 0), sample_celeba_img_fnc])
            task_func_dict['test'].extend([partial(sample_cifar10_img_fnc, 0), sample_celeba_img_fnc])          
        elif task == 'airplane':
            task_func_dict['train'].extend([partial(sample_cifar10_img_fnc, 0)])
        elif task == 'hier-imagenet':
            task_func_dict['train'] = create_hier_imagenet_supertasks(data_dir='/disk_c/han/data/ImageNet/', info_dir='./imagenet_class_hierarchy/modified', level=4)
        elif task == 'mnist':
            if len(classes) <= 0:
                classes = list(range(0, 10))
            task_func_dict['train'].extend([partial(sample_mnist_img_fnc, l) for l in classes])
        elif task == 'fashion_mnist':
            if len(classes) <= 0:
                classes = list(range(0, 10))
            task_func_dict['train'].extend([partial(sample_fashion_mnist_img_fnc, l) for l in classes])
        elif task == 'mnist_fmnist':
            if len(classes) <= 0:
                classes = list(range(0, 10))
            task_func_dict['train'].extend([partial(sample_mnist_img_fnc, l) for l in classes])
            task_func_dict['train'].extend([partial(sample_fashion_mnist_img_fnc, l) for l in classes])
        elif task == 'mnist_fmnist_3level':
            if len(classes) <= 0:
                classes = list(range(0, 10))
            task_func_dict['train'].extend([partial(sample_mnist, classes), partial(sample_fashion_mnist, classes)])
        elif task == 'LQR':
            pass
        else:
            raise Exception('Task not implemented/undefined')

    print(task_func_dict)
    return task_func_dict