import torch
import torch.nn as nn
from torch.utils.data import DataLoader  #, Subset
from torch.nn.utils import clip_grad_value_
import random
from torch.optim import Adam, SGD
from functools import partial
import numpy as np

from dataset import Meta_Dataset, Meta_Dataset_LQR, Meta_DataLoader #, get_lv0_dataset, get_high_lv_dataset #, get_samples  
from task.make_tasks_new import get_task_fnc
from utils import get_args  #, vis_img_recon
# from utils import optimize, manual_optim, send_to
# from finite_diff import debug_top
# from torch.autograd import gradcheck

from torch.nn.utils.clip_grad import clip_grad_value_


# from collections import OrderedDict
# import higher 

from pdb import set_trace
import IPython

DOUBLE_precision = False #True

print_forward_test = False
print_forward_return = False #True
print_task_loader = True
print_optimize_level_iter = False
print_optimize_level_over = True #False

##############################################################################
# set self.ctx as parameters # BAD IDEA
# implment MAML: Done
# TO DO: 
# add RNN encoder 
# add RL task & RL encoder 
# To Do: explort train and test loss: (line 55)


###################################


def get_hierarchical_task(task_name, classes, k_batch_dict, n_batch_dict):
    task_func = get_task_fnc(task_name, classes)
    task = Hierarchical_Task(task_func, (k_batch_dict, n_batch_dict), idx=0)
    return Meta_Dataset(data=[task])

def move_to_device(input_tuple, device):
    if device is None:
        return input_tuple
    else:
        return [k.to(device) for k in input_tuple]

##############################################################################
#  Model Hierarchy

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, n_contexts, max_iters, for_iters, lrs, encoders, logger, ctx_logging_levels = [], Higher_flag = False, data_parallel=False): 
        super().__init__()
        # assert hasattr  (decoder_model, 'forward')    # submodel has a built-in forward() method 

        if data_parallel:
            decoder_model = nn.DataParallel(decoder_model)
            
        self.decoder_model  = decoder_model            
        self.n_contexts = n_contexts
        self.args_dict = {'max_iters' : max_iters,
                          'for_iters' : for_iters, 
                          'lrs'       : lrs,
#                           'loggers'   : loggers,
#                           'test_loggers': test_loggers,
                          }
        self.logger = logger
        self.ctx_logging_levels = ctx_logging_levels
        self.Higher_flag = Higher_flag  # Use Higher or not
#         self.test_loggers = test_loggers
        
        print('max_iters', max_iters)
        
        self.device    = decoder_model.device
        self.level_max = len(decoder_model.parameters_all) #- 1
            
        # self.adaptation = optimize if encoder_model is None else encoder_model 

    def forward(self, task_batch, level=None, optimizer=SGD, reset=True, return_outputs=False, status='', viz=None):        
        '''
        args: minibatch of tasks 
        returns:  mean_test_loss, mean_train_loss, outputs

        Encoder(Adaptation) + Decoder model:
        Takes train_samples, adapt to them, then 
        Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks 
        '''
        if level is None:
            level = self.level_max 
        next_level = level-1
        status_lv = status+'/lv'+str(next_level)
                
        if level == 0:
            loss, outputs = self.decoder_model(move_to_device(task_batch, self.device))
            
        else:
            loss, count, outputs = 0, 0, []

            for task in task_batch:   # Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
                Flag = optimize(self, task.loader['train'], next_level, self.args_dict, self.logger, optimizer=optimizer, reset=reset, status=status_lv+str(task.idx)+'train', device=self.device, ctx_logging_levels = self.ctx_logging_levels, Higher_flag = self.Higher_flag)
                
                test_batch = next(iter(task.loader['test']))
                l, outputs_ = self(test_batch, next_level, return_outputs=return_outputs, status=status_lv+str(task.idx)+'test')      # test only 1 minibatch
                if print_forward_test:
                    print('next_level', next_level, 'test loss', l.item())
                
                loss  += l
                count += 1
                if return_outputs:
                    outputs.append(outputs_)
                    
                
            if status == viz:
                visualize_output(outputs)

            loss = loss / count
            
        self.logging(loss, status)

        if print_forward_return: # and level>0:
            print(status+'loss', loss.item()) 
        return loss, outputs
    
    def logging(self, loss, status):
        self.logger.experiment.add_scalar("loss{}".format(status), loss) #, self.iter)
#         if not self.no_print and not (self.iter % self.update_iter):
#             self.log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))

#         self.iter += 1
        

##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]


class Hierarchical_Task():
    def __init__(self, task, batch_dict, idx): 
        total_batch_dict, mini_batch_dict = batch_dict
        
        self.task = task
        self.idx = idx
        self.level = len(total_batch_dict) - 1          # print(self.level, total_batch_dict)
        self.total_batch = total_batch_dict[-1]         # total_batch: total # of samples   
        self.mini_batch = mini_batch_dict[-1]           # mini_batch: mini batch # of samples
        self.batch_dict_next = (total_batch_dict[:-1], mini_batch_dict[:-1])
        
        self.loader = {'train': self.get_dataloader(self.task, sample_type='train', idx=self.idx), 
                       'test':  self.get_dataloader(self.task, sample_type='test', idx=self.idx)}
    
        if print_task_loader:
            print('Task_loader Level', self.level, 'task', self.task)

    def get_dataloader(self, task, sample_type, idx):     #     sample_type = 'train' or 'test'  
        total_batch, mini_batch = self.total_batch[sample_type], self.mini_batch[sample_type],

        if self.level == 0:
            dataset = get_lv0_dataset(task, total_batch, sample_type) #, batch_dict_next = None)
            if mini_batch == 0 and total_batch == 0:     # To fix:  why assume total_batch == 0   ??
                mini_batch = len(dataset)                  # Full range/batch if both 0s
            shuffle = True if sample_type == 'train' else False
            return DataLoader(dataset, batch_size=mini_batch, shuffle=shuffle)                # returns tensors
        else:
            dataset = get_high_lv_dataset(task, total_batch, sample_type, self.batch_dict_next)
            return Meta_DataLoader(dataset, batch_size=mini_batch, task_name=str(task), task_idx=idx)      #  returns a mini-batch of Hiearchical Tasks[

    

##################################
def get_lv0_dataset(task, total_batch, sample_type, batch_dict_next = None):             # make a dataset out of the samples from the given task
    assert isinstance(task,tuple)

    input_generator, target_generator = task  # Generator functions for the input and target
    input_data  = input_generator(total_batch, sample_type)  # Huh : added sample_type as input
    target_data = target_generator(input_data) if target_generator is not None else None
    if DOUBLE_precision:
        input_data  = input_data.double();    
        target_data = target_data.double() if target_data is not None else None  
        
    if isinstance(input_data,tuple):
        assert len(input_data)==3
        return Meta_Dataset_LQR(data=input_data, target=target_data)
    else:
        return Meta_Dataset(data=input_data, target=target_data)
    
    
#########

def get_high_lv_dataset(task, total_batch, sample_type, batch_dict_next = None):             # make a dataset out of the samples from the given task
    if isinstance(task, list):       #  sampling from task list # To fix: task does not take sample_type as an input
        assert total_batch <= len(task)
        subtask_samples = random.sample(task, total_batch)                      
    elif callable(task):             #  sampling from task_generating function 
        subtask_samples = list(task(sample_type) for _ in range(total_batch))   
    else: 
#         set_trace()
        print(task)
        error()
    subtask_list = [Hierarchical_Task(subtask, batch_dict_next, idx) for (idx, subtask) in enumerate(subtask_samples)]  # Recursive
    return Meta_Dataset(data=subtask_list)





####################################    
def optimize(model, dataloader, level, args_dict, logger, optimizer, reset, status, device, ctx_logging_levels, Higher_flag):       
    ## optimize parameter for a given 'task'
#     lr, max_iter, for_iter, logger = get_args(args_dict, level)
    lr, max_iter, for_iter = get_args(args_dict, level)
#     task_name = dataloader.task_name if hasattr(dataloader,'task_name') else None
    task_idx = dataloader.task_idx if hasattr(dataloader,'task_idx') else None
    
    grad_clip_value = 100 #1000
    
    ####################################
    def initialize():     ## Initialize param & optim
        param_all = model.decoder_model.module.parameters_all if isinstance(model.decoder_model, nn.DataParallel) else model.decoder_model.parameters_all

        optim = None
        if param_all[level] is not None:
            if reset:  # use manual optim or higher 
                param_all[level] = torch.zeros_like(param_all[level], requires_grad=True, device=device)   # Reset
                if Higher_flag: # using higher 
                    optim = optimizer([param_all[level]], lr=lr)
                    optim = higher.get_diff_optim(optim, [param_all[level]]) #, device=x.device) # differentiable optim for inner-loop:
            else: # use regular optim: for outer-loop
                optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim
                
        return param_all, optim
    ######################################
    def update_step():
        if param_all[level] is not None:      
            if reset:  # use manual optim or higher 
                if Higher_flag:
                    param_all[level], = optim.step(loss, params=[param_all[level]])    # syntax for diff_optim
                else: # manual SGD step
                    first_order = False
                    if param_all[level+1] is None:  # for LQR_.. without network model parameters
                        first_order = True
                        
                    grad = torch.autograd.grad(loss, param_all[level], create_graph=not first_order)
                    for g in grad: #filter(lambda p: p.grad is not None, parameters):
                        g.data.clamp_(min=-grad_clip_value, max=grad_clip_value)
                    param_all[level] = param_all[level] - lr * grad[0]

            else: # use regular optim: for outer-loop
                optim.zero_grad()
                loss.backward()
                clip_grad_value_(param_all[level](), clip_value=grad_clip_value) #20)
                optim.step()  
                
    ######################################
    def log_ctx(logger, status, ctx):
        if level in ctx_logging_levels:   ##  list of levels for logging ctx variables 
            if ctx is None or ctx.numel() == 0 or logger is None:
                pass
            else:
                # Log each context changing separately if size <= 5
                if ctx.numel() <= 5:
                    for i, ctx_ in enumerate(ctx.flatten()): #range(ctx.size):
                        logger.experiment.add_scalar("Context{}/{}".format(status,i), ctx_) #, self.iter)
                else:
                    logger.experiment.add_histogram("Context {}".format(task_name), ctx) #, self.iter)

    
    ######################################
    
    param_all, optim = initialize()  
    cur_iter = 0
    loss = None
    
    while True:
        for i, task_batch in enumerate(dataloader):
            for _ in range(for_iter):          # Seungwook: for_iter is to replicate cavia’s implementation where they use the same mini-batch for the inner loop steps
                if cur_iter >= max_iter:      # Terminate after max_iter of batches/iter
                    log_ctx(logger, task_idx, param_all[level])    # log the final ctx for the level

                    if print_optimize_level_over:
                        print('optimize'+status, 'loss_f', loss.item() if loss is not None else None)
                    return False  #loss_all   # Loss-profile

                loss = model(task_batch, level, status=status)[0]     # Loss to be optimized
                        
                if print_optimize_level_iter:
                    print('level',level, 'batch', i, 'cur_iter', cur_iter, 'loss', loss.item())
                    
#                 if logger is not None:      # - logging -
#                     print('level',level, 'batch', i, 'cur_iter', cur_iter, 'loss', loss.item())
#                     logger.log_loss(loss.item(), level, num_adapt=cur_iter)
                    
                update_step()
                    
                cur_iter += 1   
#     return cur_iter 
        
