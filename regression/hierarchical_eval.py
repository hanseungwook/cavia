import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import numpy as np
from functools import partial

from optimize import optimize
from utils import move_to_device, check_nan #, get_args  #, vis_img_recon #manual_optim, send_to
# from finite_diff import debug_top
# from torch.autograd import gradcheck

from torch.nn.utils.clip_grad import clip_grad_value_

from pdb import set_trace
import IPython

print_forward_test = False
print_forward_return = False #True
# print_task_loader = True

task_merge_len = 10 #5

##############################################################################
# TO DO: 
# add RNN encoder 
# add RL task & RL encoder 
# To Do: explort train and test loss: (line 55)


##############################################################################
#  Model Hierarchy

class Hierarchical_Eval(nn.Module):            # Bottom-up hierarchy
    def __init__(self, hparams, decoder_model, encoder_model, base_loss, logger):
        
        super().__init__()

        h_names = ['n_contexts', 'max_iters', 'for_iters', 'lrs', 'log_intervals', 'test_intervals', 'log_loss_levels', 'log_ctx_levels', 'task_separate_levels', 'print_levels', 'use_higher', 'grad_clip']
        for name in h_names:
            setattr(self, name, getattr(hparams, name))

        self.decoder_model = nn.DataParallel(decoder_model) if hparams.data_parallel else decoder_model 
        self.device        = self.decoder_model.device
        self.base_loss     = base_loss
        self.logger        = logger
        # self.encoder_model = encoder_model
        # self.adaptation = optimize if encoder_model is None else encoder_model 
       

    def forward(self, level, task_list,    optimizer=SGD, reset=True, return_outputs=False, status='',  viz=None, iter_num = 0):        
        # '''   Compute average loss over multiple tasks in task_list     '''
        status = update_status(status, level = level)

        log_loss_flag   = level     in self.log_loss_levels
        log_ctx_flag    = level     in self.log_ctx_levels
        task_merge_flag = level-1 in self.task_separate_levels
        print_loss_flag = level-1 in self.print_levels 

        if level > 0:          # meta-level evaluation
            eval_fnc = partial(self.meta_eval, level = level-1, status =  status, optimizer=optimizer, reset=reset, return_outputs=return_outputs, iter_num=iter_num, task_merge_flag=task_merge_flag, print_loss_flag=print_loss_flag)
            loss, outputs = get_average_loss(eval_fnc, task_list, return_outputs) 
        else:                  # base-level evaluation (level 0)
            loss, outputs = self.base_eval(task_list)
            
        self.logging(level, loss.item(), status, iter_num, log_loss_flag = log_loss_flag, log_ctx_flag = log_ctx_flag) #, log_interval = self.log_intervals[level])
        # if status == viz:   #visualize_output(outputs)
        return loss, outputs
    
    def base_eval(self, task_list):
        # set_trace()
        if hasattr(self.decoder_model, 'name') and self.decoder_model.name == 'LQR_env':
            inputs = task_list
            loss, outputs = self.decoder_model(inputs)
        else:
            assert isinstance(task_list[0], (torch.FloatTensor, torch.DoubleTensor)) #, torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
            inputs, targets, task_idx = move_to_device(task_list, self.device)
            if hasattr(self.decoder_model, 'loss_fnc'):
                loss, outputs = self.decoder_model(inputs, targets)
            else:
                outputs = self.decoder_model(inputs)
                loss = self.base_loss(outputs, targets)
        check_nan(loss)          # assert not torch.isnan(loss), "loss is nan"
        return loss, outputs


    def meta_eval(self, task, task_idx, level, status, optimizer, reset, return_outputs, iter_num, task_merge_flag, print_loss_flag):
        # '''  Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks     '''
        status = update_status(status, task_idx=task_idx, task_merge_flag=task_merge_flag)
        def run_test(iter_num):       # evaluate on one mini-batch from test-loader
            l, outputs_ = self.forward(level, task.load('test'), return_outputs=return_outputs, status= update_status(status, sample_type='test'), iter_num=iter_num) #, current_status = 'test' + current_status)    # test only 1 minibatch
            if print_loss_flag:
                print('level',level, 'iter', iter_num, 'test loss', l.item())
            return l, outputs_

        # inner-loop optimization
        optimize(self, task.load('train'), level, self.lrs[level], self.max_iters[level], self.for_iters[level], self.test_intervals[level],
                        status = update_status(status, sample_type='train'),
                        run_test = run_test, 
                        optimizer = optimizer, 
                        reset = reset, 
                        device = self.device, 
                        Higher_flag     = self.use_higher,
                        grad_clip       = self.grad_clip)

        return run_test(iter_num=self.max_iters[level]) # return  test-loss after adaptation


    #################
    ####  Logging 

    def logging(self, level, loss, status, iter_num, log_loss_flag, log_ctx_flag, log_interval = 1):
        if not (iter_num % log_interval):
            if status != '' and log_loss_flag:
                self.log_loss(loss, status, iter_num)

           if status != '' and log_ctx_flag: #  and reset:
                decoder = self.decoder_model.module if isinstance(self.decoder_model, nn.DataParallel) else self.decoder_model
                self.log_ctx(decoder.parameters_all[level], status, iter_num, )  #log the adapted ctx  #not for the outer-loop

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
def update_status(status, sample_type = None, level = None, task_idx = None, task_merge_flag = None):
    if status is not '' and level is not None:
        status += '_lv'+str(level)
    if status is not '' and not task_merge_flag and task_idx is not None:
        status += '/' + str(task_idx)   # status += '/task' + str(task_idx) 
    if sample_type is not None:
        status += '/' + sample_type
    # print('status :', status)
    return status
