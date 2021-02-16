import torch
import torch.nn as nn
from torch.utils.data import DataLoader  #, Subset
from torch.nn.utils import clip_grad_value_
# import random
from torch.optim import Adam, SGD
from functools import partial
import numpy as np

# from dataset import Meta_Dataset, Meta_Dataset_LQR, Meta_DataLoader #, get_dataloader_dict #, get_lv0_dataset, get_high_lv_dataset #, get_samples  
# from task.make_tasks_new import get_task_fnc
from utils import get_args  #, vis_img_recon
# from utils import optimize, manual_optim, send_to
# from finite_diff import debug_top
# from torch.autograd import gradcheck

from torch.nn.utils.clip_grad import clip_grad_value_


# from collections import OrderedDict
# import higher 

from pdb import set_trace
import IPython

# DOUBLE_precision = False #True

print_forward_test = False
print_forward_return = False #True
# print_task_loader = True
print_optimize_level_iter = False
print_optimize_level_over = True #False

##############################################################################
# set self.ctx as parameters # BAD IDEA
# implment MAML: Done
# TO DO: 
# add RNN encoder 
# add RL task & RL encoder 
# To Do: explort train and test loss: (line 55)


##############################################################################
#  Model Hierarchy

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, encoders, logger, n_contexts, max_iters, for_iters, lrs, ctx_logging_levels = [], Higher_flag = False, data_parallel=False): 
        super().__init__()

        if data_parallel:
            decoder_model = nn.DataParallel(decoder_model)
            
        self.decoder_model  = decoder_model            
        self.n_contexts = n_contexts
        self.args_dict = {'max_iters' : max_iters,
                          'for_iters' : for_iters, 
                          'lrs'       : lrs,
#                           'loggers'   : loggers,
                          }
        self.logger = logger
        self.ctx_logging_levels = ctx_logging_levels
        self.Higher_flag = Higher_flag  # Use Higher ?
        
        print('max_iters', max_iters)
        
        self.device    = decoder_model.device
        self.level_max = len(decoder_model.parameters_all) #- 1

        # self.adaptation = optimize if encoder_model is None else encoder_model 

    def forward(self, task_list, level=None, optimizer=SGD, reset=True, return_outputs=False, prev_status='', current_status='', viz=None):        
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
                
        if level == 0:
            loss, outputs = self.decoder_model(move_to_device(task_list, self.device))
            
        else:
            loss, outputs = [], []

            for task in task_list:   # Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
                status_idx = prev_status + '/lv'+str(next_level)+'_task'+str(task.idx)
#                 prev_status = prev_status+current_status+'/'
#                 current_status = 'lv'+str(next_level)+'_task'+str(task.idx)
                
#                 set_trace()
                ctx_opt = optimize(self, task.load('train'), next_level, self.args_dict, 
                                    optimizer=optimizer, reset=reset, prev_status=status_idx+'/train', current_status = '' , # 'train' + current_status, 
                                    device=self.device, Higher_flag = self.Higher_flag)
    
                if level in self.ctx_logging_levels:   
                    self.log_ctx(ctx_opt, prev_status + current_status)  # log_ctx(logger, task_idx, ctx_opt)  # log the adapted ctx for the level

                # evaluate on one mini-batch from test dataloader
                l, outputs_ = self(task.load('test'), next_level, return_outputs=return_outputs, prev_status=status_idx + '/test') #, current_status = 'test' + current_status)    # test only 1 minibatch
                if print_forward_test:
                    print('next_level', next_level, 'test loss', l.item())
                
                loss.append(l)
                if return_outputs:
                    outputs.append(outputs_)
                
            loss = torch.stack(loss).mean() 
            
#             if status == viz:
#                 visualize_output(outputs)
#         if print_forward_return: # and level>0:
#             print(status+'loss', loss.item()) 
            
#         self.logging(loss, prev_status + current_status)
        return loss, outputs
    
    def logging(self, loss, status): #, name):
        self.logger.experiment.add_scalar("loss{}".format(status), loss) #, self.iter)
#         self.logger.experiment.add_scalars("loss{}".format(status), {name :loss } ) #, self.iter)
    
#         if not self.no_print and not (self.iter % self.update_iter):
#             self.log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))

#         self.iter += 1

    ######################################
    def log_ctx(self, ctx, status):   #   def log_ctx(self, prev_status, current_status, ctx):
        if ctx is None or ctx.numel() == 0 or logger is None:
            pass
        else:
            # Log each context changing separately if size <= 3
            if ctx.numel() <= 3:
                for i, ctx_ in enumerate(ctx.flatten()): #range(ctx.size):
                    self.logger.experiment.add_scalar("ctx{}/{}".format(prev_status,i), {current_status:ctx_}) #, self.iter)
            else:
                self.logger.experiment.add_histogram("ctx{}".format(prev_status), {current_status:ctx}) #, self.iter)

    

####################################   
## Optimize

def optimize(model, dataloader, level, args_dict, optimizer, reset, prev_status, current_status, device, Higher_flag):       
    ## optimize parameter for a given 'task'
    lr, max_iter, for_iter = get_args(args_dict, level)
    task_idx = dataloader.task_idx if hasattr(dataloader,'task_idx') else None
#     task_name = dataloader.task_name if hasattr(dataloader,'task_name') else None
    
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
    
    param_all, optim = initialize()  
    cur_iter = 0
    loss = None
    
    while True:
        for i, task_batch in enumerate(dataloader):
            for _ in range(for_iter):          # Seungwook: for_iter is to replicate cavia’s implementation where they use the same mini-batch for the inner loop steps
                if cur_iter >= max_iter:      # Terminate after max_iter of batches/iter

                    if print_optimize_level_over:
                        print('optimize'+prev_status+current_status, 'loss_f', loss.item() if loss is not None else None)
                    return param_all[level] #False  #loss_all   # Loss-profile

                loss = model(task_batch, level, prev_status=prev_status, current_status = current_status)[0]     # Loss to be optimized
                model.logging(loss, prev_status + current_status, cur_iter)

#                 if print_optimize_level_iter:
#                     print('level',level, 'batch', i, 'cur_iter', cur_iter, 'loss', loss.item())
                    
#                 if logger is not None:      # - logging -
#                     logger.log_loss(loss.item(), level, num_adapt=cur_iter)
                    
                update_step()
                    
                cur_iter += 1   
#     return cur_iter 
        

##############################

def move_to_device(input_tuple, device):
    if device is None:
        return input_tuple
    else:
        return [k.to(device) for k in input_tuple]