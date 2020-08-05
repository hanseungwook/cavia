import torch
import torch.nn as nn
from torch.utils.data import DataLoader  #, Subset
import random

from utils import manual_optim
from dataset import Meta_Dataset, Meta_DataLoader, get_samples  
from finite_diff import debug_lower   #debug_top
# from torch.autograd import gradcheck


# from pdb import set_trace

DOUBLE_precision = True

DEBUG_LEVELs = []  # [1] #[0]  #[2]

##############################################################################
#  Model Hierarchy

def make_hierarhical_model(model, n_contexts, n_iters, lrs, encoders, loggers):
    
    for level, (n_context, n_iter, lr, encoder, logger) in enumerate(zip(n_contexts, n_iters, lrs, encoders, loggers)):
        model = Hierarchical_Model(model, level, n_context, n_iter, lr, encoder, logger) #, adaptation_type) 
        print('level', level, '(n_context, n_iter, lr, encoder, logger)', (n_context, n_iter, lr, encoder, logger))
    return model


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

        self.adaptation = self.optimize if encoder_model is None else encoder_model 

        self.reset_ctx()

    def reset_ctx(self):
        if DOUBLE_precision:
            self.ctx = torch.zeros(1, self.n_context, requires_grad=True).double().to(self.device)
        else:
            self.ctx = torch.zeros(1, self.n_context, requires_grad=True).double().float().to(self.device)

    def forward(self, minibatch_task, ctx_high = [], optimizer = manual_optim, outerloop = False, grad_clip = None): 
        # assert self.level == tasks[0].level                                # checking if the model level matches with task level
        loss, count = 0, 0
        for task in minibatch_task:
            # print('level', self.level, 'adapt')
            self.adaptation(task.loader['train'], ctx_high, optimizer=optimizer, outerloop=outerloop, grad_clip=grad_clip)                         # adapt self.ctx  given high-level ctx 
            # print('level', self.level, 'test')
            
            for minibatch_test in task.loader['test']:
                loss += self.submodel(minibatch_test, ctx_high + [self.ctx]) [0]    # going 1 level down
                count += 1
                break                            # test only 1 minibatch

        return loss / count,  None  

    def optimize(self, dl, ctx_high, optimizer, outerloop, grad_clip):       # optimize parameter for a given 'task'
    # def optimize(self, dl, ctx_high, optimizer = manual_optim, outerloop = False, grad_clip = None, logger_update = None):       # optimize parameter for a given 'task'
        if outerloop:
            params = self.parameters()           # outer-loop parameters: model.parameters()
        else: 
            self.reset_ctx()
            params = [self.ctx]                  # inner-loop parameters: context variables 

        optim = optimizer(params, self.lr) #, grad_clip)   
        cur_iter = 0

        while True:
            for minibatch in dl:
                if cur_iter >= self.max_iter:    # train/optimize up to max_iter minibatches
                    return 
                loss = self.submodel(minibatch, ctx_high + [self.ctx]) [0] 

                if self.level in DEBUG_LEVELs: 
                    debug_lower(self.submodel, params, minibatch, loss, ctx_high, level = self.level)

                optim.zero_grad()
                if hasattr(optim, 'backward'):   # manual_optim case  # cannot call loss.backward() in inner-loops
                    optim.backward(loss)
                else:
                    loss.backward()
                optim.step()             #  check for memory leak                                                         # model.ctx[level] = model.ctx[level] - args.lr[level] * grad            # if memory_leak:
                cur_iter += 1   

            # ------------ logging ------------
            if self.logger is not None:
                self.logger.update(cur_iter, loss.detach().cpu().numpy())



##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]


def get_hierarhical_task(task_func_list, k_batch_dict, n_batch_dict):
    task = Hierarchical_Task(task_func_list, (k_batch_dict, n_batch_dict))
    return Meta_Dataset(data=[task])


class Hierarchical_Task():
    def __init__(self, task, batch_dict): 
        total_batch_dict, mini_batch_dict = batch_dict
        self.task = task
        self.level = len(total_batch_dict) - 1
        # print(self.level, total_batch_dict)
        self.total_batch = total_batch_dict[-1]
        self.mini_batch = mini_batch_dict[-1]
        self.batch_dict_next = (total_batch_dict[:-1], mini_batch_dict[:-1])
        self.loader = self.get_dataloader_dict()

    def get_dataloader_dict(self):
        return {'train': self.get_dataloader(self.total_batch['train'], self.mini_batch['train'], sample_type='train'), 
                'test':  self.get_dataloader(self.total_batch['test'],  self.mini_batch['test'],  sample_type='test')}


    # total_batch: total # of samples /  mini_batch: mini batch # of samples
    def get_dataloader(self, total_batch, mini_batch, sample_type):
        if self.level == 0:
            input_gen, target_gen = self.task
            input_data = input_gen(total_batch)
            target_data = target_gen(input_data)

            if DOUBLE_precision:
                input_data = input_data.double();  target_data=target_data.double()

            dataset = Meta_Dataset(data=input_data, target=target_data)
            return DataLoader(dataset, batch_size=mini_batch, shuffle=True)                # returns tensors

        else:
            tasks = get_samples(self.task, total_batch, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]  # recursive
            subtask_dataset = Meta_Dataset(data=subtask_list)
            return Meta_DataLoader(subtask_dataset, batch_size=mini_batch)            #   returns a mini-batch of Hiearchical Tasks[
    
