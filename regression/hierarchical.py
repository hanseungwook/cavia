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
print_optimize_level_over = False #False

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
    def __init__(self, levels, decoder_model, encoder_model, logger, n_contexts, max_iters, for_iters, lrs, loss_logging_levels = [], ctx_logging_levels = [], Higher_flag = False, data_parallel=False): 
        
        super().__init__()
        
#         levels           = len(n_contexts) + 1 #len(decoder_model.parameters_all) #- 1
        self.levels      = levels
        self.n_contexts  = n_contexts
        self.max_iters   = max_iters
        self.for_iters   = for_iters #or [1]*levels
        self.lrs         = lrs #or [0.01]*levels
#         self.args_dict      = {'max_iters' : max_iters,
#                                'for_iters' : for_iters or [1]*levels, 
#                                'lrs'       : lrs or [0.01]*levels}
        self.logger = logger
        self.loss_logging_levels = loss_logging_levels
        self.ctx_logging_levels = ctx_logging_levels
        self.Higher_flag        = Higher_flag 
        
        if data_parallel:
            decoder_model  = nn.DataParallel(decoder_model)
        self.decoder_model = decoder_model            
        self.device        = decoder_model.device
        # self.adaptation = optimize if encoder_model is None else encoder_model 
        print('max_iters', max_iters)

    def high_level_foward(self, task, level, status, optimizer, reset, return_outputs):
        status_idx = status +'_task'#+str(task.idx)  # prev_status = prev_status+current_status+'/'   # current_status = 'lv'+str(level)+'_task'+str(task.idx)

        # inner-loop optimization
        optimize(self, task.load('train'), level, self.lrs[level], self.max_iters[level], self.for_iters[level], #self.args_dict, 
                        optimizer=optimizer, reset=reset, prev_status=status_idx+'/train', current_status = '' , # 'train' + current_status, 
                        device=self.device, Higher_flag = self.Higher_flag, 
                        log_loss_flag = (level in self.loss_logging_levels),
                        log_ctx_flag = (level in self.ctx_logging_levels))

        # evaluate on one mini-batch from test dataloader
        l, outputs_ = self(task.load('test'), level, return_outputs=return_outputs, prev_status=status_idx + '/test') #, current_status = 'test' + current_status)    # test only 1 minibatch
        return l, outputs_

        

    def forward(self, task_list, level=None, optimizer=SGD, reset=True, return_outputs=False, prev_status='', current_status='', viz=None):        
        '''
        args: minibatch of tasks 
        returns:  mean_loss, outputs

        Encoder(Adaptation) + Decoder model:
        Takes train_samples, adapt to them, then 
        Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks 
        '''
        

        def get_average_loss(model, task_list):
            loss = []
            for t_ in tasks:
                l = model(t_)
                loss.append(l)
            return  torch.stack(loss).mean() 

        
        if level is None:
            level = self.levels 
        lower_level = level-1
                
        if level == 0:
            loss, outputs = self.decoder_model(move_to_device(task_list, self.device))
            
        else:
            loss, outputs = [], []

            for task in task_list:   # Todo: Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
                l, outputs_ = self.high_level_foward(task, lower_level, prev_status + '/lv'+str(lower_level), optimizer, reset, return_outputs)
                loss.append(l)
                if return_outputs:
                    outputs.append(outputs_)
            
#                 if print_forward_test:          
#                     print('lower_level', lower_level, 'test loss', l.item())

            loss = torch.stack(loss).mean() 
            
#             if status == viz:
#                 visualize_output(outputs)
        return loss, outputs
    
    ############################    
    def logging(self, loss, status, iter_num): #, name):
        self.logger.experiment.add_scalar("loss{}".format(status), loss, iter_num)   #  .add_scalars("loss{}".format(status), {name: loss}, iter_num)

    ######################################
    def log_ctx(self, ctx, status, iter_num):   #   def log_ctx(self, prev_status, current_status, ctx):
        if ctx is None or ctx.numel() == 0 or self.logger is None:
            pass
        else:
            # Log each context changing separately if size <= 3
            if ctx.numel() <= 3:
                for i, ctx_ in enumerate(ctx.flatten()): #range(ctx.size):
                    # self.logger.experiment.add_scalar("ctx{}/{}".format(prev_status,i), {current_status: ctx_}, iter_num)
                    self.logger.experiment.add_scalar("ctx{}/{}".format(status,i), ctx_, iter_num)
            else:
                self.logger.experiment.add_histogram("ctx{}".format(status), ctx, iter_num)


####################################   
## Optimize

def optimize(model, dataloader, level, lr, max_iter, for_iter, optimizer, reset, prev_status, current_status, device, Higher_flag, log_loss_flag, log_ctx_flag):       
    ## optimize parameter for a given 'task'
#     lr, max_iter, for_iter = get_args(args_dict, level)
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
                # set_trace()
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
    def logging(loss, param, cur_iter):
#         if not (cur_iter % update_iter[level]):
            if log_loss_flag:
                model.logging(loss, prev_status+current_status, cur_iter)  #log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))
            if log_ctx_flag:
                model.log_ctx(param_all[level], prev_status + current_status, cur_iter)  #  log the adapted ctx for the level
                
    ######################################
    
    param_all, optim = initialize()  
    cur_iter = 0
    loss = None
    
    while True:
        for i, task_batch in enumerate(dataloader):
            for _ in range(for_iter):          # Seungwook: for_iter is to replicate cavia’s implementation where they use the same mini-batch for the inner loop steps
                if cur_iter >= max_iter:       # Terminate! 
                    if print_optimize_level_over and loss is not None:
                        print('optimize'+prev_status+current_status, 'loss_final', loss.item())
                    return None   # param_all[level] # for log_ctx

                loss, output = model(task_batch, level, prev_status=prev_status, current_status = current_status)    # Loss to be optimized
                
                logging(loss.item(), param_all[level], cur_iter)
                if print_optimize_level_iter:
                    print('level',level, 'batch', i, 'cur_iter', cur_iter, 'loss', loss.item())
                
                update_step()
                cur_iter += 1   

##############################

def move_to_device(input_tuple, device):
    if device is None:
        return input_tuple
    else:
        return [k.to(device) for k in input_tuple]