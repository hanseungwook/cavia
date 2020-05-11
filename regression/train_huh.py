import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
from model.models_huh import get_model_type, get_encoder_type
from functools import partial
from inspect import signature
import random


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
            ENCODER_TYPE = get_encoder_type(args.model_type)
            encoder_model = ENCODER_TYPE( n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
            encoders.append(encoder_model)
    return encoders


def run(args, log, tb_writer=[]):
    base_model      = get_base_model(args)
    encoder_models  = get_encoder_model(args.encoders)

    model   = make_hierarhical_model(base_model, args.n_contexts, args.n_iters[:-1], args.lrs[:-1], encoder_models)
    dataset = Level2Dataset(args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    logger  = Logger(log, tb_writer, args.log_name, args.log_interval)

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
    def __init__(self):
        self.data = None
    
    def sample(self):
        raise NotImplementedError

# Constitutes/returns upon sample a list of Level1Dataset
# Passes parameter generating and task generating function to lower dataset
class Level2Dataset(Dataset):
    def __init__(self, *n_batch_all, num_supertasks=2):
        super().__init__()
        n_batch_train, n_batch_test, n_batch_valid = n_batch_all
        self.n_batch = {'train': n_batch_train[-1], 'test': n_batch_test[-1], 'valid': n_batch_valid[-1]}
        self.n_batch_next =     (n_batch_train[:-1],        n_batch_test[:-1],         n_batch_valid[:-1])
        self.n_super_tasks = num_supertasks
        self.level = len(self.n_batch_next[0])
    
    # Returns a np array of Level1Datasets
    def sample(self, sample_type):
        n_batch = self.n_batch[sample_type]
        assert n_batch <= self.n_super_tasks        # Cannot sample more than the total # of supertasks

        gen_fns = random.sample(task_func, n_batch)

        return [Level1Dataset(gen_fn, *self.n_batch_next) for gen_fn in gen_fns]

# Constitutes/returns upon sample a list of Level0Dataset
# Receives parameter generating and task generating function
# Passes task function (with parameters defined) to lower dataset
class Level1Dataset(Dataset):
    def __init__(self, gen_fn, *n_batch_all, num_presample=100):
        n_batch_train, n_batch_test, n_batch_valid = n_batch_all
        self.n_batch = {'train': n_batch_train[-1], 'test': n_batch_test[-1], 'valid': n_batch_valid[-1]}
        self.n_batch_next =     (n_batch_train[:-1],        n_batch_test[:-1],         n_batch_valid[:-1])

        self.num_presample = num_presample
        self.param_gen_fn, self.task_gen_fn = gen_fn
        self.train_data = None
        self.test_data = None

        self.level = len(self.n_batch_next[0])

        # Sample both train and test fn parameters
        self.sample_params()

    # Returns a np array of Level0Datasets
    def sample(self, sample_type):
        n_batch = self.n_batch[sample_type]

        if sample_type == 'train':
            return np.random.choice(self.train_data, n_batch, replace=False)
        elif sample_type == 'test':
            return np.random.choice(self.test_data, n_batch, replace=False)

    def sample_params(self):
        self.train_data = [Level0Dataset(self.task_gen_fn(*self.param_gen_fn()), *self.n_batch_next) for i in range(self.num_presample)]
        self.test_data = [Level0Dataset(self.task_gen_fn(*self.param_gen_fn()), *self.n_batch_next) for i in range(self.num_presample)]

# Returns [inputs, outputs] upon sample
# Receives task function (with parameters defined)
class Level0Dataset(Dataset):
    def __init__(self, task_fn, *n_batch_all, num_presample=100):
        n_batch_train, n_batch_test, n_batch_valid = n_batch_all
        self.n_batch = {'train': n_batch_train[-1], 'test': n_batch_test[-1], 'valid': n_batch_valid[-1]}
        self.n_batch_next =     (n_batch_train[:-1],        n_batch_test[:-1],         n_batch_valid[:-1])

        self.task_fn = task_fn
        self.train_inputs = None
        self.test_inputs = None
        self.num_presample = num_presample

        self.level = len(self.n_batch_next[0])

        self.sample_inputs()

    # Returns a list: [inputs, outputs]
    def sample(self, sample_type):
        n_batch = self.n_batch[sample_type]

        if sample_type == 'train':
            return [self.train_inputs[:n_batch], self.task_fn(self.train_inputs[:n_batch])]
        elif sample_type == 'test':
            return [self.test_inputs[:n_batch], self.task_fn(self.test_inputs[:n_batch])]

    def sample_inputs(self):
        self.train_inputs = torch.randn(self.num_presample, 1)
        self.test_inputs = torch.randn(self.num_presample, 1)

##############################################################################
#  Base Task
class Base_Task():
    def __init__(self, task_idx=None):
        #self.task_object initialize
        self.task_fn = None
        self.task_idx = task_idx
        self.params = None

        if self.task_idx is not None:
            self.params, self.task_fn = task_func(self.task_idx)
            self.task_fn = self.task_fn(*self.params)
    
    def __call__(self, x):
        # self.task object call
        assert self.task_fn is not None              # Task function has to be defined at this point
        assert self.params is not None               # Params of task function have to be defined

        return self.task_fn(x)

def get_sin_params():
    # Sample n_batch number of parameters
    amplitude = np.random.uniform(0.1, 5.)
    phase = np.random.uniform(0., np.pi)

    return amplitude, phase

def get_sin_function(amplitude, phase):
    def sin_function(x):
        if isinstance(x, torch.Tensor):
            return torch.sin(x - phase) * amplitude
        else:
            return np.sin(x - phase) * amplitude

    return sin_function

def get_linear_params():
    slope = np.random.uniform(-3., 3.)
    bias = np.random.uniform(-3., 3.)

    return slope, bias

def get_linear_function(slope, bias):
    def linear_function(x):
        return slope * x + bias

    return linear_function

def get_quadratic_params():
    slope1 = np.random.uniform(-0.2, 0.2)
    slope2 = np.random.uniform(-2.0, 2.0)
    bias = np.random.uniform(-3., 3.)

    return slope1, slope2, bias

def get_quadratic_function(slope1, slope2, bias):
    def quadratic_function(x):
        if isinstance(x, torch.Tensor):
            return slope1 * torch.pow(x, 2) + slope2 * x + bias
        else:
            return slope1 * np.squre(x, 2) + slope2 * x + bias

    return quadratic_function

def get_cubic_params():
    slope1 = np.random.uniform(-0.1, 0.1)
    slope2 = np.random.uniform(-0.2, 0.2)
    slope3 = np.random.uniform(-2.0, 2.0)
    bias = np.random.uniform(-3., 3.)

    return slope1, slope2, slope3, bias

def get_cubic_function(slope1, slope2, slope3, bias):
    def cubic_function(x):
        if isinstance(x, torch.Tensor):
            return \
                slope1 * torch.pow(x, 3) + \
                slope2 * torch.pow(x, 2) + \
                slope3 * x + \
                bias
        else:
            return \
                slope1 * np.power(x, 3) + \
                slope2 * np.power(x, 2) + \
                slope3 * x + \
                bias

    return cubic_function

task_func = [(get_sin_params, get_sin_function),
             (get_linear_params, get_linear_function),
            #  (get_cubic_params, get_cubic_function),
            #  (get_quadratic_params, get_quadratic_function)]\
            ]

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


#####################################
# Manual optim :  replace optim.SGD  due to memory leak problem

class manual_optim():
    def __init__(self, param_list, lr):
        self.param_list = param_list
        self.lr = lr

    def zero_grad(self):
        for par in self.param_list:
            par.grad = torch.zeros(par.data.shape, device=par.device)

    def backward(self, loss):
        assert len(self.param_list) == 1            # may only work for a list of one?  # shouldn't run autograd multiple times over a graph 
        for par in self.param_list:  
            par.grad += torch.autograd.grad(loss, par, create_graph=True)[0]                 # grad = torch.autograd.grad(loss, model.ctx[level], create_graph=True)[0]                 # create_graph= not args.first_order)[0]

    def step(self):
        for par in self.param_list:
            par.data = par.data - self.lr * par.grad
            # par.data -= self.lr * par.grad      # compare these
            # par = par - self.lr * par.grad      # compare these
            # par -= par.grad * self.lr           # # compare these... get tiny differences in grad result.


#####################################
class Logger():
    def __init__(self, log, tb_writer, log_name, update_iter):
        self.log = log
        self.tb_writer = tb_writer
        self.log_name = log_name
        self.update_iter = update_iter
    def update(self, iter, loss):
        if not (iter % self.update_iter):
            
            self.log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format( iter, loss))
            self.tb_writer.add_scalar("Meta loss:", loss, iter)

