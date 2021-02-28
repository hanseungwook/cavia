import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import numpy as np
from functools import partial
from torch.nn.utils.clip_grad import clip_grad_value_

from optimize import optimize
from utils import move_to_device, check_nan #,  vis_img_recon, send_to, manual_optim

from pdb import set_trace
import IPython

print_forward_test = False
print_forward_return = False

# TO DO:  add RNN encoder 

##############################################################################
###  Hierarchical Meta Evaluator 
## Summary pseudo-code ##
# 
# class Hierarchical_Eval(nn.Module):       
#     def forward(self, task_list):        
#         if level > 0:          # meta-loss
#             return get_average_loss(self.meta_eval, task_list, return_outputs) 
#         else:                  # base-loss
#             return self.base_eval(*task_list)            # task_list = (inputs, targets)

#     def meta_eval(self, task):
#         optimize(self, dataloader = task.load('train'))   # task.dataloaders['train']
#         return self.forward(task.load('test'))  # test    # next(iter(task.dataloaders['test']))

#     def base_eval(self, inputs, targets):
#         outputs = self.decoder_model(inputs)
#         loss = self.base_loss(outputs, targets)

# def get_average_loss(task_list):
#     loss =[]
#     for task in task_list:   # Todo: Parallelize!
#         loss.append( meta_eval(task, idx) )
#     return torch.stack(loss).mean()

# def optimize(model, dataloader)
#     for task_list in dataloader:  # task_list = sampled mini-batch
#         for _ in range(max_iter): 
#             loss, output = model.forward(task_list)  

##############################################################################

class Hierarchical_Eval(nn.Module):            # Bottom-up hierarchy
    def __init__(self, hparams, decoder_model, encoder_model, base_loss, logger):
        super().__init__()
        print('building Hierarchical_Environment') 

        h_names = ['top_level', 'n_contexts', 'max_iters', 'for_iters', 'lrs', 
                   'log_intervals', 'test_intervals', 'log_loss_levels', 'log_ctx_levels', 'task_separate_levels', 'print_levels', 
                   'use_higher', 'grad_clip']
        for name in h_names:
            setattr(self, name, getattr(hparams, name))

        self.decoder_model = nn.DataParallel(decoder_model) if hparams.data_parallel else decoder_model 
        self.device        = self.decoder_model.device
        self.base_loss     = base_loss
        self.logger        = logger
        # self.encoder_model = encoder_model
        # self.adaptation = optimize if encoder_model is None else encoder_model 
       
    def forward(self, task_list,  sample_type: str, level: int = None,  status: str = "", status_dict: dict = {}, 
                      optimizer=SGD, reset=True, return_outputs=False,  viz=None, iter_num = 0):        
        # To-do: fix / delete return_outputs
        # '''   Compute average loss over multiple tasks in task_list     '''
        if level is None:
            level = self.top_level

        status, status_dict = update_status(status, status_dict, sample_type=sample_type, level = level) 
        log_loss_flag, log_ctx_flag, print_flag, task_separate_flag = self.get_flags(level)
 
        if level > 0:          # meta-level evaluation
            eval_fnc = partial(self.evaluate, level = level-1, status =  status, status_dict = status_dict, optimizer=optimizer, reset=reset, return_outputs=return_outputs, iter_num=iter_num, task_separate_flag=task_separate_flag) #, print_flag=print_flag)
            loss, outputs = get_average_loss(eval_fnc, task_list, return_outputs) 
        else:                  # base-level evaluation (level 0)
            loss, outputs = self.base_eval(task_list)
            
        self.logging(level, loss.item(), status, iter_num, log_loss_flag, log_ctx_flag, self.log_intervals[level], print_flag) 
        self.visualize(outputs)
        return loss, outputs
    
    def base_eval(self, task_list):
        if self.base_loss is None: # base_loss included in decoder_model
            inputs = task_list
            loss, outputs = self.decoder_model(inputs)
        else:
            assert isinstance(task_list[0], (torch.FloatTensor, torch.DoubleTensor)) #, torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
            inputs, targets, task_idx = move_to_device(task_list, self.device)
            outputs = self.decoder_model(inputs)
            loss = self.base_loss(outputs, targets)
        check_nan(loss)  
        return loss, outputs


    def evaluate(self, task, task_idx, level, status, status_dict, optimizer, reset, return_outputs, iter_num, task_separate_flag): #, print_flag):
        # '''  adapt on train-tasks / then test the generalization loss   '''
        status, status_dict = update_status(status, status_dict, task_idx=task_idx, task_separate_flag=task_separate_flag)

        def test_eval(iter_num):       # evaluate on one mini-batch from test-loader
            l, outputs_ = self.forward(task.load('test'), sample_type='test', level=level, status = status, status_dict = status_dict, return_outputs=return_outputs, iter_num=iter_num) #, current_status = 'test' + current_status)    # test only 1 minibatch
            return l, outputs_

        # inner-loop optimization/adaptation
        optimize(self, task.load('train'), level, self.lrs[level], self.max_iters[level], self.for_iters[level], self.test_intervals[level],
                        status = status, status_dict = status_dict,
                        test_eval = test_eval, 
                        optimizer = optimizer, 
                        reset = reset, 
                        device = self.device, 
                        Higher_flag     = self.use_higher,
                        grad_clip       = self.grad_clip)

        return test_eval(iter_num=self.max_iters[level]) # return  test-loss after adaptation

    #################
    ####  flags 

    def get_flags(self, level):
        log_loss_flag   = level in self.log_loss_levels
        log_ctx_flag    = level in self.log_ctx_levels
        print_flag      = level in self.print_levels          # print_flag  = level-1 in self.print_levels # when inside evaluate
        task_separate_flag = level-1 in self.task_separate_levels    
        return log_loss_flag, log_ctx_flag, print_flag, task_separate_flag

    #################
    ####  Logging 

    def logging(self, level, loss, status, iter_num, log_loss_flag, log_ctx_flag, log_interval = 1, print_flag=False):
        if not (iter_num % log_interval):

            if log_loss_flag:
                self.log_loss(loss, status, iter_num)

            if status != '' and log_ctx_flag: #  and reset:
                decoder = self.decoder_model.module if isinstance(self.decoder_model, nn.DataParallel) else self.decoder_model
                self.log_ctx(decoder.parameters_all[level], status, iter_num, )  #log the adapted ctx  #not for the outer-loop

            if print_flag and status[-1]=='t':  #'n': # #print for 'test' samples only (not 'train' samples)
                print(status, 'iter', iter_num, 'loss', loss)
                
                # loss_status_dict = {
                #     'level1 task':  task1_idx, 
                #     sample_type
                #     'level1 iter':  task1_iter,   
                #     'level0 task':  task0_idx,  
                #     'level0 iter':  iter_num,
                # }

    def log_loss(self, loss, status, iter_num):
        self.logger.experiment.add_scalar("loss{}".format(status), loss, iter_num)   #  .add_scalars("loss{}".format(status), {name: loss}, iter_num)
        #log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))

    def log_ctx(self, ctx, status, iter_num):
        if ctx is None or callable(ctx) or ctx.numel() == 0:  #or self.logger is None:
            pass
        else:
            if ctx.numel() <= 4:   # Log each context 
                for i, ctx_ in enumerate(ctx.flatten()): #range(ctx.size):
                    self.logger.experiment.add_scalar("ctx{}/{}".format(status, i), ctx_, iter_num)
            else:                  # Log histogram of ctx
                self.logger.experiment.add_histogram("ctx{}".format(status), ctx, iter_num)


    ###########
    ### visualization
    def visualize(self, outputs):
        pass
        # if self.viz_flag:   
        #     visualize_output(outputs)

##################################
# get_average_loss - Parallelize! 

def get_average_loss(eval_fnc, task_list, return_outputs):
    loss, outputs = [], []

    for (param, task, idx) in task_list:   # Todo: Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
        l, outputs_ = eval_fnc(task, idx)
        loss.append(l)
        if return_outputs:
            outputs.append(outputs_)

    return torch.stack(loss).mean() , outputs


###########################
def update_status(status, status_dict, sample_type = None, level = None, task_idx = None, task_separate_flag = None):
    # if status is not '' and level is not None:
    if level is not None:
        status += '/lv'+str(level)         # status += '_lv'+str(level)
    if sample_type is not None:
        status += '_' + sample_type
    # if status is not '' and not task_separate_flag and task_idx is not None:
    if task_separate_flag and task_idx is not None:
        status += '/' + str(task_idx)           # status += '/task_' + str(task_idx) 
    # print('status :', status)
    return status, status_dict
