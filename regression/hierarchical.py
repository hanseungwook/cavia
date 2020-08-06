import torch
import torch.nn as nn
from torch.utils.data import DataLoader  #, Subset
import random

from utils import optimize, manual_optim, send_to
from dataset import Meta_Dataset, Meta_DataLoader, get_samples  
# from torch.autograd import gradcheck

import pdb

DOUBLE_precision = False #True


##############################################################################
# TO DO: 
# set self.ctx as parameters 
# implment MAML
# add RNN encoder 
# add RL task & RL encoder 
# To Do: explort train and test loss: (line 55)


###################################

def make_hierarhical_model(model, n_contexts, n_iters, lrs, encoders, loggers):   
    for level, (n_context, n_iter, lr, encoder, logger) in enumerate(zip(n_contexts, n_iters, lrs, encoders, loggers)):
        model = Hierarchical_Model(model, level, n_context, n_iter, lr, encoder, logger) #, adaptation_type) 
        print('level', level, '(n_context, n_iter, lr, encoder, logger)', (n_context, n_iter, lr, encoder, logger))
        if DOUBLE_precision:
            model.double()
    return model


def get_hierarhical_task(task_func_list, k_batch_dict, n_batch_dict):
    task = Hierarchical_Task(task_func_list, (k_batch_dict, n_batch_dict))
    return Meta_Dataset(data=[task])


##############################################################################
#  Model Hierarchy

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, level, n_context, n_iter, lr, encoder_model = None, logger = None): 
        super().__init__()
        assert hasattr  (decoder_model, 'forward')    # submodel has a built-in forward() method 

        self.level     = level                        #  useful for debugging/experimenting
        self.submodel  = decoder_model 
        self.logger    = logger
        # self.parameter  = list(decoder_model.modules_list[-1].parameters()) #decoder_model.parameter_list_next[-1] 
        # self.modules_list = decoder_model.modules_list[:-1]  #[self.base_modules, ctx1, ctx0]
        self.n_context = n_context
        self.max_iter  = n_iter                  # max number of iteration for adaptation/optimization 
        self.lr        = lr
        self.device    = decoder_model.device

        # self.adaptation = optimize if encoder_model is None else encoder_model 

        self.ctx = send_to(torch.empty(1, self.n_context, requires_grad=True), self.device, DOUBLE_precision)
        self.reset_ctx()

    def reset_ctx(self):
        # self.ctx.detach().fill_(0)
        self.ctx = send_to(torch.zeros(1, self.n_context, requires_grad=True), self.device, DOUBLE_precision)
        # self.ctx = torch.zeros(1, self.n_context, requires_grad=True)
        # print( self.ctx)

    def forward(self, task_batch, ctx_high = [], optimizer = manual_optim, outerloop = False, grad_clip = None): 
        '''
        args: minibatch of tasks 
        returns:  mean_test_loss, mean_train_loss, outputs

        Encoder(Adaptation) + Decoder model:
        Takes train_samples, adapt to them, then 
        Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks 
        '''

        # print('forward level', self.level)   # print('task level', task_batch[0].level)
        assert self.level == task_batch[0].level                                # checking if the model level matches with task level

        optim_args = (self.level, self.lr, self.max_iter, self.logger, None) #self.grad_clip

        test_loss,  test_count, train_loss_all = 0, 0, []
        for task in task_batch: 

            ## Todo: remove this condition:
            if outerloop:
                params = list(self.parameters())           # outer-loop parameters: model.parameters()
            else: 
                self.reset_ctx()                     # TO DO: set ctx as parameters 
                params = [self.ctx]                  # inner-loop parameters: context variables 

            ctx = ctx_high + [self.ctx]
            optimize(self.submodel, params, task.loader['train'], ctx, optimizer=optimizer, optim_args=optim_args)            # l_train = self.adaptation(self, task.loader['train'], ctx_high, optimizer=optimizer, outerloop=outerloop, grad_clip=grad_clip)   
            
            l, outputs = self.submodel(next(iter(task.loader['test'])), ctx)      # test only 1 minibatch
            test_loss  += l
            test_count += 1

        mean_test_loss = test_loss / test_count
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
            return DataLoader(dataset, batch_size=mini_batchsize, shuffle=True)                # returns tensors

        else:
            tasks = get_samples(self.task, total_batchsize, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]  # recursive
            subtask_dataset = Meta_Dataset(data=subtask_list)
            return Meta_DataLoader(subtask_dataset, batch_size=mini_batchsize)            #   returns a mini-batch of Hiearchical Tasks[


