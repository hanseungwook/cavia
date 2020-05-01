import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from model.models_huh import get_model_type
from functools import partial
from inspect import signature


def get_base_model(args):
    MODEL_TYPE = get_model_type(args.model_type)
    model = MODEL_TYPE( n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
    return model


def run(args, log, tb_writer=[]):
    base_model = get_base_model(args)
    base_task  = toy_base_task

    model = make_hierarhical_model(base_model, args.n_contexts, args.n_iters[:-1], args.lrs[:-1])
    task  = Hierarchical_Task(base_task, args.n_batch_train, args.n_batch_test, args.n_batch_valid)

    logger = Logger(log, tb_writer, args.log_name, args.log_interval)

    train(model, task, args.n_iters[-1], args.lrs[-1], logger)     # Train

    test_loss = model.evaluate( task.sample('test') )              # Test
    return test_loss


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
#                                              super-duper-task (base_task)        f(., ., .)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., b)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., a, b)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, a, b)]

class Hierarchical_Task():                      # Top-down hierarchy
    def __init__(self, base_task,  *n_batch_all):
        n_batch_train, n_batch_test, n_batch_valid = n_batch_all
        assert len(n_batch_train) == len(signature(base_task).parameters)   # base_task should take correct number of inputs
        self.base_task = base_task
        self.n_batch = {'train': n_batch_train[0], 'test': n_batch_test[0], 'valid': n_batch_valid[0]}
        self.n_batch_next =     (n_batch_train[1:],        n_batch_test[1:],         n_batch_valid[1:])
    
    def sample(self, sample_type):
        n_batch = self.n_batch[sample_type]

        if len(self.n_batch_next[0]) == 0:      # sample datapoints
            inputs = torch.randn(n_batch, 1)
            targets = self.base_task(inputs)
            return [inputs, targets]            # return a list of datapoints [inputs, targets]
        else:                                   # sample the subtask (recursive call on the class)
            params = torch.randn(n_batch)
            return [__class__(partial(self.base_task, par), *self.n_batch_next) for par in params]   # return a list of subtasks


def toy_base_task(x,a,b):
    return a + b * x


##############################################################################
#  Model Hierarchy

def make_hierarhical_model(model, n_contexts, n_iters, lrs):
    for n_context, n_iter, lr in zip(n_contexts, n_iters, lrs):
        model = Hierarchical_Model(model, n_context, n_iter, lr, adaptation_type = 'optimize') 
    return model


class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, submodel, n_context, n_iter, lr, adaptation_type):
        super().__init__()
        assert hasattr(submodel, 'evaluate')    # submodel has evaluate() method built-in
        self.submodel  = submodel             
        self.n_context = n_context
        self.n_iter    = n_iter
        self.lr        = lr
        # self.logger    = logger 
        self.device    = submodel.device
        self.adaptation_type = adaptation_type

        self.reset_ctx()

    def reset_ctx(self):
        self.ctx = torch.zeros(1,self.n_context, requires_grad = True).to(self.device)


    def evaluate(self, tasks, ctx_high = []):
        loss = 0                 
        for task in tasks:
            self.adaptation(task.sample('train'), ctx_high)                                # adapt self.ctx  given high-level ctx 
            loss += self.submodel.evaluate(task.sample('test'), ctx_high + [self.ctx])     # going 1 level down
        return loss / float(len(tasks))  


    def adaptation(self, tasks, ctx_high):      
        if self.adaptation_type == 'optimize':
            self.optimize(tasks, ctx_high)
        # elif self.adaptation_type == 'model_based':
        #     self.encoder_network (tasks, ctx_high)   # to be implemented

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
        if iter % self.update_iter:
            self.log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format( iter, loss))
            self.tb_writer.add_scalar("Meta loss:", loss, iter)

