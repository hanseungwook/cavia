import os
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import numpy as np
from functools import partial
from torch.nn.utils.clip_grad import clip_grad_value_

from optimize import optimize
from utils import move_to_device, check_nan #,  vis_img_recon, send_to, manual_optim

import shutil
from train_huh import get_latest_ckpt

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
#             return get_average_loss(self.evaluate, task_list) 
#         else:                  # base-loss
#             return self.base_eval(*task_list)            # task_list = (inputs, targets)

#     def evaluate(self, task):
#         optimize(self, dataloader = task.load('train'))   # task.dataloaders['train']
#         return self.forward(task.load('test'))  # test    # next(iter(task.dataloaders['test']))

#     def base_eval(self, inputs, targets):
#         outputs = self.decoder_model(inputs)
#         loss = self.base_loss(outputs, targets)

# def get_average_loss(eval, task_list):
#     loss =[]
#     for task in task_list:   # Todo: Parallelize!
#         loss.append( eval(task, idx) )
#     return torch.stack(loss).mean()

# def optimize(model, dataloader)
#     for task_list in dataloader:  # task_list = sampled mini-batch
#         for _ in range(max_iter): 
#             loss, output = model.forward(task_list)  

##############################################################################

class Hierarchical_Eval(nn.Module):            # Bottom-up hierarchy
    def __init__(self, hparams, decoder_model, encoder_model, base_loss, logger, best_loss = None):
        super().__init__()

        h_names = ['top_level', 'n_contexts', 'max_iters', 'for_iters', 'lrs', 
                   'log_intervals', 'test_intervals', 'log_loss_levels', 'log_ctx_levels', 'task_separate_levels', 'print_levels', 
                   'use_higher', 'grad_clip', 'mp',
                   'save_dir']
        for name in h_names:
            setattr(self, name, getattr(hparams, name))

        self.decoder_model = nn.DataParallel(decoder_model) if hparams.data_parallel else decoder_model 
        self.device        = self.decoder_model.device
        self.base_loss     = base_loss
        self.logger        = logger
        self.best_loss     = best_loss
        # self.encoder_model = encoder_model
        # self.adaptation = optimize if encoder_model is None else encoder_model 
       
    def forward(self, task_list,  sample_type: str, iter_num: int, level: int,  status: str, status_dict: dict, return_outputs=False):    #    reset=True
        # To-do: fix / delete return_outputs
        status, status_dict = update_status(status, status_dict, sample_type=sample_type) 
            
        log_loss_flag, log_ctx_flag, print_flag, task_separate_flag = self.get_flags(level)
 
        if level > 0:          # meta-level evaluation
            eval_fnc = partial(self.evaluate, level = level-1, status =  status, status_dict = status_dict, return_outputs=return_outputs, task_separate_flag=task_separate_flag)
            loss, outputs = get_average_loss(eval_fnc, task_list, return_outputs, mp=self.mp) 
        else:                  # base-level evaluation (level 0)
            loss, outputs = self.base_eval(task_list)
            
        self.logging(level, loss.item(), status, iter_num, log_loss_flag, log_ctx_flag, self.log_intervals[level], print_flag) 
        # self.visualize(outputs)
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


    def evaluate(self, task, task_idx, level = None, status="", status_dict={}, optimizer = SGD,  reset=True, return_outputs = False, iter0 = 0, task_separate_flag = True, optimizer_state_dict = None): #, print_flag):
        # '''  adapt on train-tasks / then test the generalization loss   '''
        if level is None:
            level = self.top_level

        status, status_dict = update_status(status, status_dict, level=level, task_idx=task_idx, task_separate_flag=task_separate_flag)

        def forward_wrapper(task_list, sample_type, iter_num):       # evaluate on one mini-batch from test-loader
            return self.forward(task_list, sample_type=sample_type, iter_num=iter_num, level=level, status = status, status_dict = status_dict, return_outputs=return_outputs)

        return optimize(self, forward_wrapper, task, self.optim_args(level),
                        optimizer = optimizer, optimizer_state_dict = optimizer_state_dict, 
                        reset = reset, 
                        device = self.device, 
                        iter0 = iter0,
                        Higher_flag     = self.use_higher,
                        grad_clip       = self.grad_clip)

    #################
    ####  flags 
    def optim_args(self, level):
        return level, self.lrs[level], self.max_iters[level], self.for_iters[level], self.test_intervals[level]

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

    ############
    def save_cktp(self, epoch, test_loss, train_loss = None, optimizer = None):
    # def save_cktp(self, train_loss, test_loss, epoch, optimizer):
        if self.best_loss is None or test_loss < self.best_loss:
            prev_file = get_latest_ckpt(self.save_dir)
            self._del_model(prev_file)

            filename = 'epoch='+str(epoch)+'.ckpt'  #  epoch=99-step=70299.ckpt
            self.best_loss = test_loss
            torch.save({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'model_state_dict': self.decoder_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                        }, os.path.join(self.save_dir,'checkpoints',filename))


    def _del_model(self, filepath):
        if filepath is None:
            pass
        else:
            dirpath = os.path.dirname(filepath)
            # make paths
            os.makedirs(dirpath, exist_ok=True)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)
                
    ###########
    ### visualization
    def visualize(self, outputs):
        pass
        # if self.viz_flag:   
        #     visualize_output(outputs)

##################################
# get_average_loss - Parallelize! 
from multiprocessing import Process, Manager

def get_average_loss(eval_fnc, task_list, return_outputs, mp=False):
    losses, outputs = [], []
    if not mp:
        for (param, task, idx) in task_list:   # Todo: Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
            l, outputs_ = eval_fnc(task, idx)
            losses.append(l)
            if return_outputs:
                outputs.append(outputs_)

        return torch.stack(losses).mean() , outputs

    else:
        manager = Manager()
        losses = manager.list()   # Strange name: please fix  @Seungwook
        process_list = []
        
        def mp_loss_fn(eval_fnc, task, idx):
            l, _ = eval_fnc(task, idx)
            losses.append(l)

        for (param, task, idx) in task_list:   # Todo: Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
            p = Process(target=mp_loss_fn, args=(eval_fnc, task, idx,))
            process_list.append(p)
            p.start()

        for p in process_list:
            p.join()

        losses = list(losses)
        return torch.stack(losses).mean() , outputs


###########################
def update_status(status, status_dict, sample_type = None, level = None, task_idx = None, task_separate_flag = None):
    # if status is not '' and level is not None:
    if level is not None:
        status += '/lv'+str(level)         # status += '_lv'+str(level)
    if task_idx is not None:
        if task_separate_flag:
            status +=  '_'+str(task_idx)           # status += '/task_' + str(task_idx) 
        # else:
        #     status += '/_'
    if sample_type is not None:
        status += '/' + sample_type
    return status, status_dict
