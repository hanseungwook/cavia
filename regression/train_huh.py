import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F

from model.models_huh import get_model_type

from functools import partial
import pdb


### Each 'task' object is assumed to have built-in sample() methods, which returns a 'list of subtasks':
# lv 2: task = supertask-set,     subtasks  = supertasks
# lv 1: task = supertask,         subtasks  = tasks (functions)
# lv 0: task = task (function),   subtasks  = (input, output) data-points


def get_base_model(args):
    n_arch = args.architecture
    n_context = sum(args.n_contexts)  #args.n_context_params

    MODEL_TYPE = get_model_type(args.model_type)
    model = MODEL_TYPE( n_arch=n_arch, n_context=n_context, device=args.device).to(args.device)
    return model


def run(args, log, tb_writer=[]):

    task = get_toy_task(args)       # implement a hierarchical task here
    base_model = get_base_model(args)
    model = make_hierarhical_model(base_model, args.n_contexts, args.n_iters[:-1], args.lrs[:-1])

    optim = Adam(model.parameters(), args.lrs[-1])
    logger = Logger(log, tb_writer, args.log_name, args.log_interval)

    for iter in range(args.n_iters[-1]): 
        loss = model.evaluate( task.sample('train') )
        optim.zero_grad()
        loss.backward()
        optim.step()

        # ------------ logging ------------
        logger.update(iter, loss.detach().cpu().numpy())
        # vis_pca(higher_contexts, task_family, iteration, args)      ## Visualize result


##############################################################################

def get_toy_task(args):    # example hierarhical task
    fnc = poly_fnc  
    task = Example_Hierarchical_Task(fnc, args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    return task


class Example_Hierarchical_Task():
    def __init__(self, fnc,  *n_batch_all):
        n_batch_train, n_batch_test, n_batch_valid = n_batch_all
        self.n_batch = {'train': n_batch_train[0], 'test': n_batch_test[0], 'valid': n_batch_valid[0]}
        self.n_batch_next =     (n_batch_train[1:],        n_batch_test[1:],         n_batch_valid[1:])
        self.f = fnc
    
    def sample(self, sample_type):
        n_batch = self.n_batch[sample_type]

        if len(self.n_batch_next[0]) == 0:   # sample datapoints
            x = torch.randn(n_batch, 1)
            y = self.f(x)
            return [x, y]                    # return a list of datapoints [inputs, targets]
        
        else:                                # sample the subtask (recursive call on the class)
            params = torch.randn(n_batch)
            return [__class__(partial(self.f,par), *self.n_batch_next) for par in params]   # return a list of subtasks


def poly_fnc(x,a,b):
    return a + b * x


##############################################################################

def make_hierarhical_model(model, n_contexts, n_iters, lrs):
    for n_context, n_iter, lr in zip(n_contexts, n_iters, lrs):
        model = Hierarchical_Model(model, n_context, n_iter, lr, adaptation_type = 'optimize') 
    model.top_level = True
    return model


class Hierarchical_Model(nn.Module):
    def __init__(self, sub_model, n_context, n_iter, lr, adaptation_type = 'optimize', logger = None):
        super().__init__()
        # sub_model has a built-in evaluate function
        self.sub_model = sub_model
        self.adaptation_type = adaptation_type
        self.ctx   = None

        self.n_context = n_context
        self.n_iter    = n_iter
        self.lr        = lr
        self.logger    = logger 
        self.device = sub_model.device

        self.reset_ctx()

    def reset_ctx(self):
        self.ctx = torch.zeros(1,self.n_context, requires_grad = True).to(self.device)


    def evaluate(self, tasks, ctx_high = []):
        loss = 0                     # going 1 level down
        for task in tasks:
            self.adaptation(task.sample('train'), ctx_high)      # adapt self.ctx    # sample subtasks_train
            loss += self.sub_model.evaluate(task.sample('test'), ctx_high + [self.ctx]) 

        return loss / float(len(tasks))  


    def adaptation(self, tasks, ctx_high):      
        if self.adaptation_type == 'optimize':
            self.optimize (tasks, ctx_high)
        # elif self.adaptation_type == 'model_based':
        #     self.encoder_network (tasks, ctx_high)   # to be implemented

    def optimize(self, tasks, ctx_high):                            # optimize parameter for a given 'task'

        self.reset_ctx()
        optim = manual_optim([self.ctx], self.lr)                   # manual optim.SGD.  check for memory leak

        for iter in range(self.n_iter): 

            loss = self.sub_model.evaluate(tasks, ctx_high + [self.ctx])  

            optim.zero_grad()
            optim.backward(loss)
            optim.step()             #  check for memory leak                                                         # model.ctx[level] = model.ctx[level] - args.lr[level] * grad            # if memory_leak:


    # def forward(self, input, ctx_high = []]):
    #     return self.sub_model(data, ctx_high + [self.ctx])            # assuming ctx is optimized over training tasks already



#####################################

# Manual optim :  replace optim.SGD  for memory leak problem
# manual_optim = optim.SGD

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
            # par.data -= self.lr * par.grad
            # par = par - self.lr * par.grad
            # par -= par.grad * self.lr   # this is bad...


class Logger():
    def __init__(self, log, tb_writer, log_name, update_iter):
        self.log = log
        self.tb_writer = tb_writer
        self.log_name = log_name
        self.update_iter = update_iter
    def update(self, iter, loss):
        if iter % self.update_iter:
            self.log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format( iter, loss))
            self.tb_writer.add_scalar("Meta loss:", loss, iter)



#########################

# class Optimizer_Encoder():
#     def __init__(self, args, logger = None):
#         self.args = args
#         self.lr = args.lr[level]
#         self.n_iter = args.n_iter[level]
#         self.top_level = args.top_level  #True if level == args.top_level else False
#         self.logger = logger

#     def step(self, loss, level):
#         if level == self.top_level:  #self.top_level:  
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#         else:
#             # for param in self.params:
#             params[0].grad = torch.autograd.grad(loss, params[0], create_graph=True)[0]                 # grad = torch.autograd.grad(loss, model.ctx[level], create_graph=True)[0]                 # create_graph= not args.first_order)[0]
#             optim.step()             #  check for memory leak                                       # model.ctx[level] = model.ctx[level] - args.lr[level] * grad            # if memory_leak:

#     def optimize(model, level, tasks):

#         if level == self.top_level:  
#             params = model.parameters()
#             optim = Adam(params, self.lr)
#         else: 
#             params = [model.ctx[level]]
#             optim = manual_optim(params, self.lr)             # replacing optim.SGD .  check for memory leak

#         for iter in range(self.n_iter): 
#             loss = evaluate(self.args, model, level, tasks)  
#             self.step(loss, level)

#             # ------------ logging ------------
#             if iter % args.logger & logger is not None:
#                 logger.update(iter, loss.detach().cpu().numpy())




#########################################
# Pseudo-code
 
# def evaluate(level = 0, tasks = datapoints = [datapoint_i])       : fit each datapoint  (for given function)
# return loss = F.mse_loss(datapoints)  

# def optimize(level = 0):
#     for loop:
#         loss = evaluate(level = 0)
#         backward_step()


# def evaluate(level = 1,  tasks = fncs = [fnc_i])                : fit each function  (for given function_type e.g. 'sinusoid')
# for fnc_i in tasks:
#     sample train_datapoints_i, test_datapoints_i
#     phi0_i = optimize(level = 0,  task = train_datapoints_i)     # optimize phi0_i   
#     loss_i = evaluate(level = 0,  task = test_datapoints_i)      # using    phi0_i  
# return sum(loss_i)

# def optimize(level = 1, task):
#     for loop:
#         loss = evaluate(level = 1, task)
#         backward_step()

# def evaluate(level = 2, tasks = fnc_types = [fnc_type_i])       : fit each function_type  (e.g. 'sinusoid')
# for fnc_type_i in tasks:
#     sample train_fncs_i, test_datapoints_i
#     phi1_i = optimize(level = 1,  task = train_fncs_i)       # optimize phi1_i   
#     loss_i = evaluate(level = 1,  task = test_fncs_i)        # using    phi1_i  
# return sum(loss_i)

# def optimize(level = 2,  task = ALL_TASK):      # ALL_TASK = [function_types]
#     for loop:
#         loss = evaluate(level = 2, task)  # 1 task
#         backward_step()

# theta = optimize(level = 2,  tasks = [function_types])       # optimize theta = phi2 




