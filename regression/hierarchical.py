import torch
import torch.nn as nn
from torch.utils.data import DataLoader  #, Subset
from torch.nn.utils import clip_grad_value_
import random
from torch.optim import Adam, SGD
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# from utils import optimize, manual_optim, send_to
from dataset import Meta_Dataset, Meta_DataLoader, get_samples  
from task.mixture2 import sample_sin_fnc, sample_linear_fnc, sample_celeba_img_fnc, sample_cifar10_img_fnc, create_hier_imagenet_supertasks, img_size
from utils import vis_img_recon, get_args
from finite_diff import debug_top
# from torch.autograd import gradcheck

import higher 

from pdb import set_trace
import IPython

DOUBLE_precision = False #True


##############################################################################
# set self.ctx as parameters # BAD IDEA
# implment MAML: Done
# TO DO: 
# add RNN encoder 
# add RL task & RL encoder 
# To Do: explort train and test loss: (line 55)


###################################

def make_tasks(task_names):
    task_func_dict = {}
    for task in task_names:
        if task == 'sine':
            task_func_dict['train'] = [sample_sin_fnc]
            # task_func_list.append(sample_sin_fnc)
        elif task == 'linear':
            task_func_dict['train'] = [sample_linear_fnc]
            # task_func_list.append(sample_linear_fnc)
        elif task == 'celeba':
            task_func_dict['train'] = [sample_celeba_img_fnc]
            # task_func_list.append(sample_celeba_img_fnc)
        elif task == 'cifar10':
            # task_func_dict['train'] = [partial(sample_cifar10_img_fnc, l) for l in range(5)]
            # task_func_dict['test'] = [partial(sample_cifar10_img_fnc, l) for l in range(5,10)]
            task_func_dict['train'] = [partial(sample_cifar10_img_fnc, l) for l in range(7)]
            task_func_dict['test'] = [partial(sample_cifar10_img_fnc, l) for l in range(7,10)]
        elif task == 'celeba_airplane':
            task_func_dict['train'] = [partial(sample_cifar10_img_fnc, 0), sample_celeba_img_fnc]
            task_func_dict['test'] = [partial(sample_cifar10_img_fnc, 0), sample_celeba_img_fnc]            

            # for l in range(1):
            #     task_func_list.append(partial(sample_cifar10_img_fnc, l))
        elif task == 'airplane':
            task_func_dict['train'] = [partial(sample_cifar10_img_fnc, 0)]
        elif task == 'hier-imagenet':
            task_func_dict['train'] = create_hier_imagenet_supertasks(data_dir='/disk_c/han/data/ImageNet/', info_dir='./imagenet_class_hierarchy/modified', level=4)
            # task_func_list = create_hier_imagenet_supertasks(data_dir='/disk_c/han/data/ImageNet/', info_dir='./imagenet_class_hierarchy/modified', level=4)
        else:
            raise Exeption('Task not implemented/undefined')

    return task_func_dict

def get_hierarchical_task(task_list, k_batch_dict, n_batch_dict):
    task_func_list = make_tasks(task_list)
    task = Hierarchical_Task(task_func_list, (k_batch_dict, n_batch_dict))
    return Meta_Dataset(data=[task])


##############################################################################
#  Model Hierarchy

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, n_contexts, max_iters, for_iters, lrs, encoders, loggers, test_loggers): 
        super().__init__()
        # assert hasattr  (decoder_model, 'forward')    # submodel has a built-in forward() method 

        self.decoder_model  = decoder_model 
        self.n_contexts = n_contexts
        self.args_dict = {'max_iters' : max_iters,
                          'for_iters' : for_iters, 
                          'lrs'       : lrs,
                          'loggers'   : loggers,
                          'test_loggers': test_loggers
                          }
        self.device     = decoder_model.device

        # self.adaptation = optimize if encoder_model is None else encoder_model 

        self.level_max = len(decoder_model.parameters_all) #2

    def forward(self, task_batch, level=None, optimizer=SGD, reset=True, return_outputs=False, status='', viz=None):        # def forward(self, task_batch, ctx_high = [], optimizer = manual_optim, outerloop = False, grad_clip = None): 
        '''
        args: minibatch of tasks 
        returns:  mean_test_loss, mean_train_loss, outputs

        Encoder(Adaptation) + Decoder model:
        Takes train_samples, adapt to them, then 
        Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks 
        '''

        if level is None:
            level = self.level_max
        
        # print(status)

        # assert level == task_batch[0].level + 1                 # check if the level matches with task level        # print('level', level , task_batch[0].level  )

        if level == 0:
            inputs, targets = task_batch
            task_batch = [inputs.to(self.device), targets.to(self.device)]
            return self.decoder_model(task_batch)
        else:
            test_loss,  test_count = 0, 0

            for task in task_batch: 
                Flag = optimize(self, task.loader['train'], level-1, self.args_dict, optimizer=optimizer, reset=reset, status=status+'train', device=self.device)
                test_batch = next(iter(task.loader['test']))
                l, outputs = self(test_batch, level-1, return_outputs=return_outputs, status=status+'test')      # test only 1 minibatch
                self.args_dict['test_loggers'][level-1].log_loss(l) # Update test logger for respective level
                test_loss  += l
                test_count += 1
                
            if status == viz:
                img_pred = outputs.view(img_size).detach().numpy()
                img_pred = np.clip(img_pred, 0, 1)
                plt.imshow(img_pred)
                plt.show()


            mean_test_loss = test_loss / test_count
            # Propagate outputs back with full batch

            if not return_outputs:
                outputs = None

            return mean_test_loss, outputs


##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]


class Hierarchical_Task():
    def __init__(self, task, batch_dict): 
        total_batch_dict, mini_batch_dict = batch_dict
        self.task = task
        self.level = len(total_batch_dict) - 1          # print(self.level, total_batch_dict)
        self.total_batch = total_batch_dict[-1]
        self.mini_batch = mini_batch_dict[-1]
        self.batch_dict_next = (total_batch_dict[:-1], mini_batch_dict[:-1])
        self.loader = self.get_dataloader_dict()

    def get_dataloader_dict(self):
        return {'train': self.get_dataloader(self.total_batch['train'], self.mini_batch['train'], sample_type='train'), 
                'test':  self.get_dataloader(self.total_batch['test'],  self.mini_batch['test'],  sample_type='test')}

    # total_batch: total # of samples  //  mini_batch: mini batch # of samples
    def get_dataloader(self, total_batchsize, mini_batchsize, sample_type):
        if self.level == 0:
            input_gen, target_gen = self.task
            input_data  = input_gen(total_batchsize)
            target_data = target_gen(input_data)

            if DOUBLE_precision:
                input_data = input_data.double();  target_data=target_data.double()

            dataset = Meta_Dataset(data=input_data, target=target_data)

            # Full range/batch if both 0s
            if mini_batchsize == 0 and total_batchsize == 0:
                mini_batchsize = len(dataset)

            shuffle = True if sample_type == 'train' else False

            return DataLoader(dataset, batch_size=mini_batchsize, shuffle=shuffle)                # returns tensors


        else:
            tasks = get_samples(self.task, total_batchsize, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]  # recursive
            subtask_dataset = Meta_Dataset(data=subtask_list)
            return Meta_DataLoader(subtask_dataset, batch_size=mini_batchsize, task_name=str(self.task))            #   returns a mini-batch of Hiearchical Tasks[


####################################    
def optimize(model, dataloader, level, args_dict, optimizer, reset, status, device):       # optimize parameter for a given 'task'
    lr, max_iter, for_iter, logger = get_args(args_dict, level)
    param_all = model.decoder_model.parameters_all

    ## Initialize param & optim
    if reset:
        param_all[level] = torch.zeros_like(param_all[level], requires_grad=True, device=device)   # Reset
        optim = optimizer([param_all[level]], lr=lr)
        optim = higher.get_diff_optim(optim, [param_all[level]]) #, device=x.device) # differentiable optim for inner-loop:
    else:
        optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim

    cur_iter = 0; 
    while True:
        for task_batch in dataloader:
            for _ in range(for_iter):
                # if level == 0:
                #     inputs, targets = task_batch
                #     task_batch = [inputs.to(device), targets.to(device)]

                loss = model(task_batch, level, status=status)[0]     # Loss to be optimized

                if cur_iter >= max_iter:    # train/optimize up to max_iter # of batches
                    if level == 1:
                        logger.log_ctx(dataloader.task_name, param_all[level].detach().cpu().numpy())   
                    return False #loss_all   # Loss-profile

                ## loss.backward() & optim.step()
                if reset:
                    new_param, = optim.step(loss, params=[param_all[level]])   # syntax for diff_optim
                    param_all[level] = new_param
                    
                else:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()  

                cur_iter += 1   

                # ------------ logging ------------
                if logger is not None:
                    logger.log_loss(loss.detach().cpu().numpy(), level, cur_iter)

         

            # return cur_iter  # completed  the batch


#######################################


