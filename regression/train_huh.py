import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import random
# import numpy as np
from functools import partial
from inspect import signature

from model.models_huh import get_model_type, get_encoder_type
from task.mixture2 import task_func_list
from utils import manual_optim


def get_base_model(args):
    MODEL_TYPE = get_model_type(args.model_type)
    model = MODEL_TYPE( n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
    return model

def get_encoder_model(encoder_types):
    encoders = []
    for encoder_type in encoder_types:
        if encoder_type is None:
            encoders.append(None)
        else:
            raise NotImplementedError()
            # ENCODER_TYPE = get_encoder_type(args.model_type)
            # encoder_model = ENCODER_TYPE( n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
            # encoders.append(encoder_model)
    return encoders


def run(args, logger):
    base_model      = get_base_model(args)
    encoder_models  = get_encoder_model(args.encoders)

    model   = make_hierarhical_model(base_model, args.n_contexts, args.n_iters[:-1], args.lrs[:-1], encoder_models)
    dataset = Level2Dataset(task_func_list, args.n_batch_train, args.n_batch_test, args.n_batch_valid)

    train(model, dataset, args.n_iters[-1], args.lrs[-1], logger)     # Train
    test_loss = model.evaluate(dataset.sample('test'))              # Test
    return test_loss, logger


def train(model, task, n_iter, lr, logger):
    
    optim = Adam(model.parameters(), lr)

    for iter in range(n_iter): 
        loss = model.evaluate( task.sample('train') )
        optim.zero_grad()
        loss.backward()
        optim.step()
        # ------------ logging ------------
        logger.update(iter, loss.detach().cpu().numpy())
        # vis_pca(higher_contexts, task_family, iteration, args)      ## Visualize result


##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]

# class Hierarchical_Task():                      # Top-down hierarchy
#     def __init__(self, dataset, *n_batch_all):
#         n_batch_train, n_batch_test, n_batch_valid = n_batch_all

#         self.n_batch = {'train': n_batch_train[-1], 'test': n_batch_test[-1], 'valid': n_batch_valid[-1]}
#         self.n_batch_next =     (n_batch_train[:-1],        n_batch_test[:-1],         n_batch_valid[:-1])
#         self.level = len(self.n_batch_next[0])
#         self.n_super_tasks = 4
#         self.dataset = dataset
    
#     def sample(self, sample_type):
#         n_batch = self.n_batch[sample_type]

#         if self.level == 0:                     # sample datapoints for the bottom level   [inputs, targets]
#             return self.dataset.sample(n_batch, sample_type)
#         elif self.level == 1:
#             return [self.__class__(self.dataset.sample(), *self.n_batch_next) for i in range(n_batch)]

# Base class for datasets (may not really be necessary)
class Dataset():
    def __init__(self, task_fn, *n_batch_all):
        self.task_fn = task_fn
        n_batch_train, n_batch_test, n_batch_valid = n_batch_all
        self.n_batch = {'train': n_batch_train[-1], 'test': n_batch_test[-1], 'valid': n_batch_valid[-1]}
        self.n_batch_next =     (n_batch_train[:-1],        n_batch_test[:-1],         n_batch_valid[:-1])
        self.level = len(self.n_batch_next[0])
        self.K_presample = {'train': 100, 'test': 100, 'valid': 100}

    def run_pre_sample(self):
        return {'train': self.pre_sample(self.K_presample['train']), 
                'test':  self.pre_sample(self.K_presample['test'])}

    def sample(self, sample_type):
        n_batch = self.n_batch[sample_type]
        if self.level == 0:
            input_data  = self.data[sample_type][:n_batch]
            target_data = self.task_fn(input_data)
            return [input_data, target_data]
        else:
            return random.sample(self.data[sample_type], n_batch)


# num_presample = [100, 100, 2]

class Level2Dataset(Dataset):
    def __init__(self, task_fn, *n_batch_all): #, num_supertasks=4):
        super().__init__(task_fn, *n_batch_all)
        self.K_presample = {'train': 2, 'test': 2, 'valid': 2}
        self.data = self.run_pre_sample()

    def pre_sample(self, K_batch):
        # assert K_batch <= self.n_super_tasks        # Cannot sample more than the total # of supertasks
        gen_fns = random.sample(self.task_fn, K_batch)
        return [Level1Dataset(gen_fn, *self.n_batch_next) for gen_fn in gen_fns]

class Level1Dataset(Dataset):
    def __init__(self, task_fn, *n_batch_all):
        super().__init__(task_fn, *n_batch_all)
        self.data = self.run_pre_sample()

    def pre_sample(self, K_batch):
        param_gen_fn, task_gen_fn = self.task_fn
        return [Level0Dataset(task_gen_fn(*param_gen_fn()), *self.n_batch_next) for i in range(K_batch)]

class Level0Dataset(Dataset):
    def __init__(self, task_fn, *n_batch_all):
        super().__init__(task_fn, *n_batch_all)
        self.data = self.run_pre_sample()

    def pre_sample(self, K_batch):
        return torch.randn(K_batch, 1)



##############################################################################
#  Model Hierarchy

def make_hierarhical_model(model, n_contexts, n_iters, lrs, encoders):
    for level, (n_context, n_iter, lr, encoder) in enumerate(zip(n_contexts, n_iters, lrs, encoders)):
        model = Hierarchical_Model(model, level, n_context, n_iter, lr, encoder) #, adaptation_type) 
    return model


class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, level, n_context, n_iter, lr, encoder_model = None): 
        super().__init__()
        assert hasattr  (decoder_model, 'evaluate')    # submodel has evaluate() method built-in
        self.submodel  = decoder_model 
        self.level     = level                  # could be useful for debugging/experimenting
        self.n_context = n_context
        self.n_iter    = n_iter
        self.lr        = lr
        self.device    = decoder_model.device

        self.adaptation = self.optimize if encoder_model is None else encoder_model 
        self.reset_ctx()

    def reset_ctx(self):
        self.ctx = torch.zeros(1,self.n_context, requires_grad = True).to(self.device)

    def evaluate(self, tasks, ctx_high = []):
        assert self.level == tasks[0].level                                               # checking if the model level matches with task level
        loss = 0 
        for task in tasks:
            self.adaptation(task.sample('train'), ctx_high)                                # adapt self.ctx  given high-level ctx 
            loss += self.submodel.evaluate(task.sample('test'), ctx_high + [self.ctx])     # going 1 level down
        return loss / float(len(tasks))  

    def optimize(self, tasks, ctx_high):                            # optimize parameter for a given 'task'
        self.reset_ctx()
        optim = manual_optim([self.ctx], self.lr)                   # manual optim.SGD.  check for memory leak

        for iter in range(self.n_iter): 
            loss = self.submodel.evaluate(tasks, ctx_high + [self.ctx])  
            optim.zero_grad()
            optim.backward(loss)
            optim.step()             #  check for memory leak                                                         # model.ctx[level] = model.ctx[level] - args.lr[level] * grad            # if memory_leak:


    def forward(self, input, ctx_high = []):                                        # assuming ctx is optimized over training tasks already
        return self.submodel(input, ctx_high + [self.ctx])            

