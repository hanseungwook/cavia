import torch
import torch.nn as nn
from torch.utils.data import DataLoader  #, Subset
from torch.nn.utils import clip_grad_value_
import random
from torch.optim import Adam, SGD
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from dataset import Meta_Dataset, Meta_DataLoader #, get_samples  
from task.make_tasks import get_task_fnc
from task.image_reconstruction import img_size
from utils import get_args  #, vis_img_recon
# from utils import optimize, manual_optim, send_to
# from finite_diff import debug_top
# from torch.autograd import gradcheck


from collections import OrderedDict
import higher 

from pdb import set_trace
import IPython

DOUBLE_precision = False #True

print_forward_test = False
print_forward_return = True
print_task_loader = True
print_optimize_level_iter = False
print_optimize_level_over = False

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
    task = Hierarchical_Task(task_func, (k_batch_dict, n_batch_dict))
    return Meta_Dataset(data=[task])

def move_to_device(input_tuple, device):
    if device is None:
        return input_tuple
    else:
        return [k.to(device) for k in input_tuple]

##############################################################################
#  Model Hierarchy

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, n_contexts, max_iters, for_iters, lrs, encoders, loggers, test_loggers, data_parallel=False): 
        super().__init__()
        # assert hasattr  (decoder_model, 'forward')    # submodel has a built-in forward() method 

        if data_parallel:
            decoder_model = nn.DataParallel(decoder_model)
            
        self.decoder_model  = decoder_model            
        self.n_contexts = n_contexts
        self.args_dict = {'max_iters' : max_iters,
                          'for_iters' : for_iters, 
                          'lrs'       : lrs,
                          'loggers'   : loggers,
                          'test_loggers': test_loggers
                          }
        
        print('max_iters', max_iters)
        
        self.device    = decoder_model.device
        self.level_max = len(decoder_model.parameters_all)
            
        # self.adaptation = optimize if encoder_model is None else encoder_model 

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
            
        if level == 0:
            loss, outputs = self.decoder_model(move_to_device(task_batch, self.device))
            
        else:
            loss, count, outputs = 0, 0, []

            for task in task_batch: 
                Flag = optimize(self, task.loader['train'], level-1, self.args_dict, optimizer=optimizer, reset=reset, status=status+'train', device=self.device)
                
                test_batch = next(iter(task.loader['test']))
                l, outputs_ = self(test_batch, level-1, return_outputs=return_outputs, status=status+'test')      # test only 1 minibatch
                if print_forward_test:
                    print('level', level, 'test loss', l.item())
                
                loss  += l
                count += 1
                if return_outputs:
                    outputs.append(outputs_)
                
                if self.args_dict['test_loggers'][level-1] is not None:
                    self.args_dict['test_loggers'][level-1].log_loss(l) # Update test logger for respective level
                
            if status == viz:
                visualize_output(outputs)

            loss = loss / count

        if print_forward_return:
            print('level', level, 'loss', loss.item()) 
        return loss, outputs

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
        self.total_batch = total_batch_dict[-1]         # total_batch: total # of samples   
        self.mini_batch = mini_batch_dict[-1]           # mini_batch: mini batch # of samples
        self.batch_dict_next = (total_batch_dict[:-1], mini_batch_dict[:-1])
        
        self.loader = {'train': self.get_dataloader(self.task, sample_type='train'), 
                       'test':  self.get_dataloader(self.task, sample_type='test')}
    
        if print_task_loader:
            print('Task_loader Level', self.level, 'task', self.task)

    def get_dataloader(self, task, sample_type):     #     sample_type = 'train' or 'test'  
        total_batch, mini_batch = self.total_batch[sample_type], self.mini_batch[sample_type],

        if self.level > 0:
            dataset = self.get_dataset(task, total_batch, sample_type, self.batch_dict_next)
            return Meta_DataLoader(dataset, batch_size=mini_batch, task_name=str(task))      #  returns a mini-batch of Hiearchical Tasks[
        else:
            dataset = self.get_dataset(task, total_batch, sample_type) #, batch_dict_next = None)
            if mini_batch == 0 and total_batch == 0:     # To fix:  why assume total_batch == 0   ??
                mini_batch = len(dataset)                  # Full range/batch if both 0s
            shuffle = True if sample_type == 'train' else False
            return DataLoader(dataset, batch_size=mini_batch, shuffle=shuffle)                # returns tensors

    def get_dataset(self, task, total_batch, sample_type, batch_dict_next = None):             # make a dataset out of the samples from the given task
        if isinstance(task, tuple): 
            assert(self.level == 0)
            assert(len(task) == 2)
            
            input_gen, target_gen = task
            input_data  = input_gen(total_batch, sample_type)  # Huh : added sample_type as input
            target_data = target_gen(input_data) if target_gen is not None else None
            if DOUBLE_precision:
                input_data  = input_data.double();    target_data = target_data.double() if target_data is not None else None        
            return Meta_Dataset(data=input_data, target=target_data)

        else:
            if isinstance(task, list):    # To fix: task does not take sample_type as an input
                assert total_batch <= len(task)
                subtask_samples = random.sample(task, total_batch)                 # sampling from list 
            else:
                subtask_samples = [task(sample_type) for _ in range(total_batch)]  # sampling from function 

            subtask_list = [self.__class__(subtask, batch_dict_next) for subtask in subtask_samples]  # Recursive
            return Meta_Dataset(data=subtask_list)

    

####################################    
def optimize(model, dataloader, level, args_dict, optimizer, reset, status, device, ctx_logging_levels = [], Higher_flag=False):       # optimize parameter for a given 'task'
    lr, max_iter, for_iter, logger = get_args(args_dict, level)
    task_name = dataloader.task_name if hasattr(dataloader,'task_name') else None
    
    ####################################
    def initialize():     ## Initialize param & optim
        # Seungwook: Why decoder_model.module? can u please comment here? 
        param_all = model.decoder_model.module.parameters_all if isinstance(model.decoder_model, nn.DataParallel) else model.decoder_model.parameters_all
        # params_all = OrderedDict(model.named_parameters())

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
                    grad = torch.autograd.grad(loss, param_all[level], create_graph=not first_order)
                    param_all[level] = param_all[level] - lr * grad[0]

            else: # use regular optim: for outer-loop
                optim.zero_grad()
                loss.backward()
                optim.step()  
                
    ######################################
    def log_ctx(logger, task_name, ctx):
        if level in ctx_logging_levels:   ##  list of levels for logging ctx variables 
            if ctx.numel() == 0 or logger is None:
                pass
            else:
                logger.log_ctx(task_name, ctx.detach().squeeze().cpu().numpy())   

        if print_optimize_level_over:
            print('optimize_level', level, 'is over')
        
    ######################################
    
    param_all, optim = initialize()  
    cur_iter = 0; 
    while True:
        for i, task_batch in enumerate(dataloader):
#             for _ in range(for_iter):          # Huh: what's for_iter for? Seungwook?
                loss = model(task_batch, level, status=status)[0]     # Loss to be optimized

                if print_optimize_level_iter:
                    print('level',level, 'batch', i, 'cur_iter', cur_iter, 'loss', loss.item())
                    
                if cur_iter >= max_iter:      # Terminate after max_iter of batches/iter
                    log_ctx(logger, task_name, param_all[level])    # log the final ctx for the level
                    return False  #loss_all   # Loss-profile

                update_step()
                        
                if logger is not None:      # - logging -
                    logger.log_loss(loss.item(), level, num_adapt=cur_iter)

                cur_iter += 1   
#     return cur_iter 
        

                
#######################################




def visualize_output(outputs, show = True, save = False):
    img_pred = outputs.view(img_size).detach().numpy()
    img_pred = np.clip(img_pred, 0, 1)                         # Forcing all predictions beyond image value range into (0, 1)
#     img_pred = np.round(img_pred * 255.0).astype(np.uint8)
#     img_pred = transforms.ToPILImage(mode='L')(img_pred)
    if show:
        plt.imshow(img_pred)
        plt.show()
    if save:
        save_dir, filename = get_filename(args)
        img_pred.save(os.path.join(save_dir, filename))        
        
    