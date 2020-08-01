import torch
import torch.nn as nn
from torch.utils.data import DataLoader  #, Subset

from utils import manual_optim
from dataset import Meta_Dataset, Meta_DataLoader  
# from finite_diff import debug_top, debug_lower
# from torch.autograd import gradcheck

import random




##############################################################################
#  Model Hierarchy

def make_hierarhical_model(model, n_contexts, n_iters, lrs, encoders):
    
    for level, (n_context, n_iter, lr, encoder) in enumerate(zip(n_contexts, n_iters, lrs, encoders)):
        model = Hierarchical_Model(model, level, n_context, n_iter, lr, encoder) #, adaptation_type) 
        print(level, (n_context, n_iter, lr, encoder))
    return model


class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, level, n_context, n_iter, lr, encoder_model = None): 
        super().__init__()
        assert hasattr  (decoder_model, 'forward')    # submodel has a built-in forward() method 

        self.level     = level                        #  useful for debugging/experimenting
        self.submodel  = decoder_model 
        # self.parameter  = list(decoder_model.modules_list[-1].parameters()) #decoder_model.parameter_list_next[-1] 
        # self.modules_list = decoder_model.modules_list[:-1]  #[self.base_modules, ctx1, ctx0]
        self.n_context = n_context
        self.n_iter    = n_iter
        self.lr        = lr
        self.device    = decoder_model.device

        self.adaptation = self.optimize if encoder_model is None else encoder_model 

        self.reset_ctx()

    def reset_ctx(self):
        if DOUBLE_precision:
            self.ctx = torch.zeros(1, self.n_context, requires_grad=True).double().to(self.device)
        else:
            self.ctx = torch.zeros(1, self.n_context, requires_grad=True).double().float().to(self.device)

    def forward(self, minibatch_task, ctx_high = [], optimizer = manual_optim, reset = True, grad_clip = None, logger_update = None): 
        # assert self.level == tasks[0].level                                               # checking if the model level matches with task level
        loss = 0
        count = 0
        for task in minibatch_task:
            # print('level', self.level, 'adapt')
            self.adaptation(task.loader['train'], ctx_high, optimizer=optimizer, reset=reset, grad_clip=grad_clip, logger_update=logger_update)                         # adapt self.ctx  given high-level ctx 
            
            # print('level', self.level, 'test')
            for minibatch_test in iter(task.loader['test']):
                loss += self.submodel(minibatch_test, ctx_high + [self.ctx])     # going 1 level down
                count += 1
                break # break after 1 minibatch

        return loss / count #float(len(minibatch))

    def optimize(self, dl, ctx_high, optimizer = manual_optim, reset = True, grad_clip = None, logger_update = None):       # optimize parameter for a given 'task'
        if reset:
            self.reset_ctx()
            params = [self.ctx]                  # inner-loop parameters: context variables 
        else: 
            params = self.parameters()           # outer-loop parameters: model.parameters()
        optim = optimizer(params, self.lr) #, grad_clip)   

        cur_iter = 0
        while True:
            for minibatch in iter(dl):
                if cur_iter >= self.n_iter: #max_iter:
                    return 
                loss = self.submodel(minibatch, ctx_high + [self.ctx])  

                # if self.level in DEBUG_LEVEL: 
                #     debug_lower(loss, params, self.submodel, minibatch, ctx_high, self.ctx, level = self.level)

                optim.zero_grad()
                if hasattr(optim, 'backward'):
                    optim.backward(loss)
                else:
                    loss.backward()
                optim.step()             #  check for memory leak                                                         # model.ctx[level] = model.ctx[level] - args.lr[level] * grad            # if memory_leak:
                cur_iter += 1   

            # ------------ logging ------------
            if logger_update is not None:
                logger_update(cur_iter, loss.detach().cpu().numpy())



##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]

# class Hierarchical_Task_TOP():
#     def __init__(self, task): 
#         self.dataset = Meta_Dataset(data=[task])
#         self.dataloader =   Meta_DataLoader(self.dataset, batch_size=1)

DOUBLE_precision = True

class Hierarchical_Task():
    def __init__(self, task, batch_dict): 
        total_batch_dict, mini_batch_dict = batch_dict
        self.task = task
        self.level = len(total_batch_dict) - 1
        # print(self.level, total_batch_dict)
        self.total_batch = total_batch_dict[-1]
        self.mini_batch = mini_batch_dict[-1]
        self.batch_dict_next = (total_batch_dict[:-1], mini_batch_dict[:-1])
        self.loader = self.run_pre_sample()

    def run_pre_sample(self):
        return {'train': self.pre_sample(self.total_batch['train'], self.mini_batch['train'], sample_type='train'), 
                'test':  self.pre_sample(self.total_batch['test'],  self.mini_batch['test'],  sample_type='test')}

    def high_level_presampling(self, total_batch, sample_type):
        if isinstance(self.task, list):
            assert total_batch <= len(self.task)
            tasks = random.sample(self.task, total_batch)
        else:
            tasks = [self.task(sample_type) for _ in range(total_batch)]
        return tasks

    # total_batch: total # of samples
    # mini_batch: mini batch # of samples
    def pre_sample(self, total_batch, mini_batch, sample_type):
        if self.level == 0:
            input_gen, target_gen = self.task
            input_data = input_gen(total_batch)
            target_data = target_gen(input_data)

            if DOUBLE_precision:
                input_data = input_data.double();  target_data=target_data.double()

            dataset = Meta_Dataset(data=input_data, target=target_data)
            return DataLoader(dataset, batch_size=mini_batch, shuffle=True)                # returns tensors

        else:
            tasks = self.high_level_presampling(total_batch, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]  # recursive
            subtask_dataset = Meta_Dataset(data=subtask_list)
            return Meta_DataLoader(subtask_dataset, batch_size=mini_batch)            #   returns a mini-batch of Hiearchical Tasks[
    
    # def get(self, sample_type):
    #     return self.dataloader[sample_type]
         

