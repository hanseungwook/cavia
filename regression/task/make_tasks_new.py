from functools import partial
from .regression_1d import sample_sin_fnc, sample_linear_fnc
from .image_reconstruction import sample_celeba_img_fnc, sample_cifar10_img_fnc, sample_mnist_img_fnc, sample_fashion_mnist_img_fnc, create_hier_imagenet_supertasks
from .LQR import LQR_environment, Linear_Policy, Combine_NN_ENV
import numpy as np

def get_task_fnc(task_names, classes):
    task_func_dict = {}
    task_func_dict['train'] = []
    task_func_dict['test'] = []
    
    assert(len(task_names) == 1) # Huh: no need to use for loop.
    # Huh: modifying code to return a function instead of dictionary task_func_dict.
    for task in task_names:
        
        if task == 'sine':
            def sample_fnc(sample_type):
                return sample_sin_fnc   # no difference between train and test
            return sample_fnc
            
        elif task == 'linear':
            def sample_fnc(sample_type):
                return sample_linear_fnc   # no difference between train and test
            return sample_fnc
        
        elif task == 'LQR':
            x_range = 4
            def sample_LQR_LV2(sample_type):                         #level 2 - dynamics
                kbm =  np.stack([np.random.uniform(0,0), np.random.uniform(-1., 1.),  np.random.uniform(-1., 1.)], axis=0)
                print('Lv2', 'sample_type', sample_type, ' kbm', kbm.shape) 
            
                def sample_LQR_LV1(sample_type):                     #level 1 - targets
                    target   = x_range * np.random.randn(1)
#                     print('Lv1', 'sample_type', sample_type, ' target', target)

                    def sample_LQR_LV0(batch_size, sample_type):      #level 0 - initial  x0
                        pos0   = x_range * np.random.randn(batch_size)
#                         print('Lv0', 'sample_type', sample_type, 'pos0', pos0)
                        task_env = kbm, target, pos0
                        return task_env
                    return sample_LQR_LV0, None    # returning input_sampler (and target_sampler = None) for level0
                return sample_LQR_LV1
            return sample_LQR_LV2 
            
############ Testing new code #######################

        elif task == 'cifar10':
            def sample_fnc(sample_type):
                if sample_type == 'train':
                    return [partial(sample_cifar10_img_fnc, l) for l in range(0,7)]
                else: 
                    return [partial(sample_cifar10_img_fnc, l) for l in range(7,10)]
            return sample_fnc
#             task_func_dict['train'].extend([partial(sample_cifar10_img_fnc, l) for l in range(7)])
#             task_func_dict['test'].extend([partial(sample_cifar10_img_fnc, l) for l in range(7,10)])

#############################################################################################################            
            
            ## HUH: Here's (a minor) consistency problem of using dictionary.  
#             'task_func' should be the Level2 function of sample_type, not a dictionary of sample_type
#             or task_func_dict can be represented as a function


        elif task == 'celeba':
            task_func_dict['train'].append(sample_celeba_img_fnc)

        elif task == 'celeba_airplane':
            task_func_dict['train'].extend([partial(sample_cifar10_img_fnc, 0), sample_celeba_img_fnc])

            # for l in range(1):
            #     task_func_list.append(partial(sample_cifar10_img_fnc, l))
        elif task == 'airplane':
            task_func_dict['train'].extend([partial(sample_cifar10_img_fnc, 0)])
        elif task == 'hier-imagenet':
            task_func_dict['train'] = create_hier_imagenet_supertasks(data_dir='/disk_c/han/data/ImageNet/', info_dir='./imagenet_class_hierarchy/modified', level=4)
            # task_func_list = create_hier_imagenet_supertasks(data_dir='/disk_c/han/data/ImageNet/', info_dir='./imagenet_class_hierarchy/modified', level=4)
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
        else:
            raise Exception('Task not implemented/undefined')

#     print(task_func_dict)
    
#     if isinstance(task_func_dict, dict):
#         if len(task_func_dict['test']) == 0:
#             task_func_dict['test'] = task_func_dict['train']
#         return dict_2_fnc(task_func_dict)
#     else:
#         return task_func_dict


# def dict_2_fnc(dict_):
#     def fnc(sample_type):
#         return dict_[sample_type]
#     return fnc