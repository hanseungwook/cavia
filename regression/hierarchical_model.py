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
# import IPython/

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

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self,  #levels, 
                        decoder_model, encoder_model, base_loss, logger, 
                        n_contexts, max_iters, for_iters, lrs, 
                        test_intervals,
                        log_loss_levels, log_ctx_levels, task_separate_levels,
                        Higher_flag = False, data_parallel=False): 
        
        super().__init__()

        if data_parallel:
            decoder_model  = nn.DataParallel(decoder_model)
        self.decoder_model = decoder_model            
        self.encoder_model = encoder_model
        self.base_loss = base_loss
        # self.adaptation = optimize if encoder_model is None else encoder_model 
        self.logger = logger

        # self.level_max   = levels
        self.n_contexts  = n_contexts
        self.max_iters   = max_iters
        self.for_iters   = for_iters
        self.lrs         = lrs 
        self.test_intervals = test_intervals

        self.log_loss_levels = log_loss_levels
        self.log_ctx_levels  = log_ctx_levels
        self.task_separate_levels     = task_separate_levels
        self.Higher_flag         = Higher_flag 
        self.device        = decoder_model.device
        

    def forward(self, level, task_list, 
                optimizer=SGD, reset=True, return_outputs=False, status='', #current_status='', 
                viz=None, iter_num = None):        
        '''
        Compute average loss over multiple tasks in task_list
        returns:  mean_loss, outputs

        Encoder(Adaptation) + Decoder model:
        '''
                
        if status is not '':
            status += '_lv'+str(level)

        if level > 0:      # meta-level evaluation
            task_merge_flag = (level-1) in self.task_separate_levels #[level-1] #len(task_list) > task_merge_len
            meta_eval_partial = partial(self.meta_eval, level = level-1, status =  status, optimizer=optimizer, reset=reset, return_outputs=return_outputs, iter_num=iter_num, task_merge_flag=task_merge_flag)
            loss, outputs = get_average_loss(meta_eval_partial, task_list, return_outputs) 

        else:                  # base-level evaluation (level 0)
            if hasattr(self.decoder_model, 'name') and self.decoder_model.name == 'LQR_env':
                inputs = task_list
                loss, outputs = self.decoder_model(inputs)
            else:
                assert isinstance(task_list[0], (torch.FloatTensor, torch.DoubleTensor)) #, torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
                inputs, targets, _ = move_to_device(task_list, self.device)
                if hasattr(self.decoder_model, 'loss_fnc'):
                    loss, outputs = self.decoder_model(inputs, targets)
                else:
                    outputs = self.decoder_model(inputs)
                    loss = self.base_loss(outputs, targets)

        check_nan(loss)          # assert not torch.isnan(loss), "loss is nan"
        
        if status is not '' and (level in self.log_loss_levels):
            self.log_loss(loss.item(), status, iter_num)              #log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))
        
        # if status == viz:   #visualize_output(outputs)

        return loss, outputs
    

    def meta_eval(self, task, task_idx, level, status, optimizer, reset, return_outputs, iter_num, task_merge_flag):
        # '''  Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks     '''
        if not task_merge_flag and status is not '':
            status += '/task' + str(task_idx)  # current_status = 'lv'+str(level)+'_task'+str(task.idx)

        def run_test(iter_num):       # evaluate on one mini-batch from test-loader
            sample_type = 'test'
            l, outputs_ = self.forward(level, task.load(sample_type), return_outputs=return_outputs, status=status+'/'+sample_type, iter_num=iter_num) #, current_status = 'test' + current_status)    # test only 1 minibatch
            return l, outputs_

        # inner-loop optimization
        sample_type = 'train' 
        optimize(self, task.load(sample_type), level,
                        self.lrs[level], self.max_iters[level], self.for_iters[level], #self.args_dict, 
                        optimizer = optimizer, reset = reset, 
                        status = status+'/'+sample_type, 
                        # current_status =  'train' + current_status, 
                        run_test = run_test, test_interval = self.test_intervals[level],
                        device = self.device, Higher_flag = self.Higher_flag, 
                        # log_loss_flag = (level in self.log_loss_levels),
                        log_ctx_flag = (level in self.log_ctx_levels))

        test_loss_optim = run_test(iter_num=self.max_iters[level]) # final test-loss
        return test_loss_optim

        
    #################
    ####  Logging 

    def log_loss(self, loss, status, iter_num): #, name):
        self.logger.experiment.add_scalar("loss{}".format(status), loss, iter_num)   #  .add_scalars("loss{}".format(status), {name: loss}, iter_num)

    def log_ctx(self, ctx, status, iter_num):   #   def log_ctx(self, status, current_status, ctx):
        if ctx is None or ctx.numel() == 0 or self.logger is None:
            pass
        else:      # Log each context changing separately if size <= 3
            if ctx.numel() <= 3:
                for i, ctx_ in enumerate(ctx.flatten()): #range(ctx.size):
                    self.logger.experiment.add_scalar("ctx{}".format(status,i), ctx_, iter_num)
                    # self.logger.experiment.add_scalar("ctx{}/{}".format(status,i), {current_status: ctx_}, iter_num)
            else:
                self.logger.experiment.add_histogram("ctx{}".format(status), ctx, iter_num)


##################################
# get_average_loss - Parallelize! 

def get_average_loss(model, task_list, return_outputs):
    loss, outputs = [], []

    for (param, task, idx) in task_list:   # Todo: Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
        l, outputs_ = model(task, idx)
        loss.append(l)
        if return_outputs:
            outputs.append(outputs_)

    return torch.stack(loss).mean() , outputs



