import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from torch.optim import Adam, SGD

# from utils import optimize, manual_optim, send_to
from dataset import Meta_Dataset, Meta_DataLoader, get_samples  
# from torch.autograd import gradcheck

import higher 

from pdb import set_trace

DOUBLE_precision = False #True


##############################################################################
# set self.ctx as parameters # BAD IDEA
# implment MAML: Done
# TO DO: 
# add RNN encoder 
# add RL task & RL encoder 
# To Do: explort train and test loss: (line 55)


###################################



def get_hierarchical_task(task_func_list, k_batch_dict, n_batch_dict):
    task = Hierarchical_Task(task_func_list, (k_batch_dict, n_batch_dict))
    return Meta_Dataset(data=[task])


##############################################################################
#  Model Hierarchy

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, n_contexts, max_iters, lrs, encoders, loggers): 
        super().__init__()
        # assert hasattr  (decoder_model, 'forward')    # submodel has a built-in forward() method 

        self.decoder_model  = decoder_model 
        self.n_contexts = n_contexts
        self.args_dict = {'max_iters' : max_iters,
                          'lrs'       : lrs,
                          'loggers'   : loggers,
                          }
        self.device     = decoder_model.device

        # self.adaptation = optimize if encoder_model is None else encoder_model 

        self.level_max = len(decoder_model.parameters_all)  #2

    def forward(self, task_batch, level = None, optimizer = SGD, reset = True):        # def forward(self, task_batch, ctx_high = [], optimizer = manual_optim, outerloop = False, grad_clip = None): 
        '''
        args: minibatch of tasks 
        returns:  mean_test_loss, mean_train_loss, outputs

        Encoder(Adaptation) + Decoder model:
        Takes train_samples, adapt to them, then 
        Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks 
        '''

        if level is None:
            level = self.level_max

        # assert level == task_batch[0].level + 1                 # check if the level matches with task level        # print('level', level , task_batch[0].level  )

        if level == 0:
            return self.decoder_model(task_batch)
        else:
            test_loss,  test_count = 0, 0
            for task in task_batch: 

                Flag = optimize(self, task.loader['train'], level-1, self.args_dict, optimizer=optimizer, reset = reset)
                test_batch = next(iter(task.loader['test']))
                l, outputs = self(test_batch, level-1)      # test only 1 minibatch
                test_loss  += l;       test_count += 1

            mean_test_loss = test_loss / test_count
            outputs = None

            # if level == self.level_max - 1: #in [2,3]:
            #     print('level', level, mean_test_loss.item())

            return mean_test_loss, outputs 


##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]


class Hierarchical_Task(object):
    def __init__(self, task, batch_dict): 
        total_batch_dict, mini_batch_dict = batch_dict
        self.task = task
        self.level = len(total_batch_dict) - 1          # print(self.level, total_batch_dict)
        self.total_batch = total_batch_dict[-1]
        self.mini_batch = mini_batch_dict[-1]
        self.batch_dict_next = (total_batch_dict[:-1], mini_batch_dict[:-1])
        self.loader = self.get_dataloader_dict()

    def get_dataloader_dict(self):
        return {
            'train': self.get_dataloader(self.total_batch['train'], self.mini_batch['train'], sample_type='train'), 
            'test':  self.get_dataloader(self.total_batch['test'], self.mini_batch['test'], sample_type='test')}

    # total_batch: total # of samples  //  mini_batch: mini batch # of samples
    def get_dataloader(self, total_batchsize, mini_batchsize, sample_type):
        if self.level == 0:
            input_gen, target_gen = self.task
            input_data  = input_gen(total_batchsize)
            target_data = target_gen(input_data)

            if DOUBLE_precision:
                input_data = input_data.double();  target_data=target_data.double()

            dataset = Meta_Dataset(data=input_data, target=target_data)
            return DataLoader(dataset, batch_size=mini_batchsize, shuffle=True)                # returns tensors

        else:
            print("self.task:", self.task)
            tasks = get_samples(self.task, total_batchsize, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]  # recursive
            subtask_dataset = Meta_Dataset(data=subtask_list)
            return Meta_DataLoader(subtask_dataset, batch_size=mini_batchsize)            #   returns a mini-batch of Hiearchical Tasks[


####################################    
def optimize(model, dataloader, level, args_dict, optimizer, reset):       # optimize parameter for a given 'task'
    lr, max_iter, logger = get_args(args_dict, level)
    param_all = model.decoder_model.parameters_all

    ## Initialize param & optim
    if reset:
        param_all[level] = torch.zeros_like(param_all[level], requires_grad= True)   # Reset
        optim = optimizer([param_all[level]], lr=lr)
        optim = higher.get_diff_optim(optim, [param_all[level]]) #, device=x.device) # differentiable optim for inner-loop:
    else:
        optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim

    cur_iter = 0; 
    while True:
        for task_batch in dataloader:
            if cur_iter >= max_iter:    # train/optimize up to max_iter # of batches
                return False #loss_all   # Loss-profile

            loss = model(task_batch, level) [0]     # Loss to be optimized

            ## loss.backward() & optim.step()
            if reset:
                new_param, = optim.step(loss, params=[param_all[level]])   # syntax for diff_optim
                param_all[level] = new_param
            else:
                optim.zero_grad()
                loss.backward()
                # if level == debug_level: # == 2:
                #     grad_list = list(par.grad for par in optim.param_groups[0]['params'])
                #     debug(model, data, level, list(param_all[level]()), grad_list)
                optim.step()  

            cur_iter += 1   

            # ------------ logging ------------
            if logger is not None:
                logger.update(cur_iter, loss.detach().cpu().numpy())

        # return cur_iter  # completed  the batch

#######################################

def get_args(args_dict, level):
    # return (arg[name][level] for arg, name in args_dict.items())
    lr = args_dict['lrs'][level] 
    max_iter = args_dict['max_iters'][level] 
    logger = args_dict['loggers'][level] 
    return lr, max_iter, logger
