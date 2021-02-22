import numpy as np
import random
from functools import partial

from .regression_1d import sample_sin_fnc, sample_linear_fnc
from .image_reconstruction_new import sample_mnist_img_fnc, sample_fmnist_img_fnc
# from .image_reconstruction import sample_celeba_img_fnc, sample_cifar10_img_fnc, create_hier_imagenet_supertasks
from .LQR import sample_LQR_LV2, sample_LQR_LV1, sample_LQR_LV0 

from pdb import set_trace



##########################

class Task_sampler():
    sample_types = ['train', 'test', 'valid']

    def __init__(self, tasks: list, 
                       k_batches: list = None, #[], 
                       task_gen_fnc = None):
        ### k_batches is a list of k_batch for the sample types
        if k_batches is None or k_batches == []:
            k_batches = [len(tasks),0,0]
        batch_cumsum = np.cumsum([0]+k_batches)
        
#         if isinstance(tasks,int):     # if 'tasks' == total number of tasks 
#             tasks = list(range(tasks))  # list of task indices/labels 
            
        self.task_gen_fnc = task_gen_fnc  # function that takes task label and returns actual tasks 
        self.task_dict={}
        self.split(tasks, batch_cumsum)
        
    def split(self, label_list, batch_cumsum):
        batch_total = batch_cumsum[-1]
        assert isinstance(label_list,list) 
        assert batch_total <= len(label_list)
        
        task_sublist = random.sample(label_list, batch_total)   
        for i, sample_type in enumerate(self.sample_types): #['train', 'test', 'valid']):
            self.task_dict[sample_type] = task_sublist[batch_cumsum[i]:batch_cumsum[i+1]]
            
        for sample_type in ['test']: #, 'valid']:             # if k_batch_test is zero, then copy 'train' task
            if len(self.task_dict[sample_type]) == 0:
                self.task_dict[sample_type] = self.task_dict['train']
            
    def sample(self, sample_type):
        assert sample_type in self.sample_types

        tasks = self.task_dict[sample_type]
        if self.task_gen_fnc is None:
            return tasks
        else:
            return [self.task_gen_fnc(l) for l in tasks] 
        
        

# subtask_list = random.sample(task, total_batch)    #  sampling from task list # To fix: task does not take sample_type as an input
# subtask_list = [task(sample_type) for _ in range(total_batch)]  #


##########
# def sample_label_mnist(sample_type):
# #     available_labels = range(0,10)
#     if sample_type = 'train':
#         sampled_labels = range(0,7)
#     else:
#         sampled_labels = range(7,10)
#     return [partial(sample_mnist_img_fnc, l) for l in sampled_labels]

sample_label_mnist = Task_sampler(tasks = list(range(10)), k_batches = [7,3,0], task_gen_fnc = sample_mnist_img_fnc).sample,
sample_label_fmnist = Task_sampler(tasks = list(range(10)), k_batches = [7,3,0], task_gen_fnc = sample_fmnist_img_fnc).sample,

# def sample_dataset_mnist_fmnist_lv3(sample_type):
#     fnc_dict = {
#         'mnist': sample_label_mnist,
#         'fmnist': sample_label_fmnist,        
#         'celeba': sample_label_celeba,        
#     }
    
#     if sample_type = 'train':
#         ds_names = ['mnist', 'fmnist']
#     else:
#         ds_names = ['mnist', 'fmnist']
# #         ds_names = ['celeba']
        
#     return [fnc_dict[name] for name in ds_names]


###########

# classes = list(range(0, 10))

get_task_dict={
    'sine':   sample_sin_fnc,
    'linear': sample_linear_fnc,
    'sine+linear': Task_sampler(tasks = [sample_sin_fnc, sample_linear_fnc]).sample,
#     
    'LQR_lv2': sample_LQR_LV2,
    'LQR_lv1': sample_LQR_LV1,
    'LQR_lv0': (sample_LQR_LV0, None),
#     
    'mnist'  : sample_label_mnist, 
    'fmnist' : sample_label_fmnist, #[partial(sample_fmnist_img_fnc, l) for l in classes],
    'mnist+fmnist' : Task_sampler(tasks = [sample_label_mnist, sample_label_fmnist]).sample,
}


# def lv2_fnc(sample_type, dataset):
#     if sample_type = 'train':
#         labels = range(0,7)
#     else:
#         labels = range(7,10)
#     return labels

# def lv1_fnc(sample_type, label):
#     sample_imange()
#     return image
