import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.autograd import gradcheck
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data import Subset, DataLoader
import random
# import numpy as np
from functools import partial
from inspect import signature
import IPython

from model.models_huh import get_model_type, get_encoder_type
from task.mixture2 import task_func_list
from utils import manual_optim
from dataset import Level0_Dataset, HighLevel_Dataset, HighLevel_DataLoader


def get_base_model(args):
    MODEL_TYPE = get_model_type(args.model_type)
    model = MODEL_TYPE( n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
    return model

def get_encoder_model(encoder_types, args):
    encoders = []
    for encoder_type in encoder_types:
        if encoder_type is None:
            encoders.append(None)
        else:
            ENCODER_TYPE = get_encoder_type(encoder_type) # Check if n_hidden is the task embedding dimension in Encoder_Core().
            encoder_model = ENCODER_TYPE( input_dim=args.input_dim, n_hidden=args.n_hidden, tree_hidden_dim=args.tree_hidden_dim, 
                                cluster_layer_0=args.cluster_layer_0, cluster_layer_1=args.cluster_layer_1)
            encoders.append(encoder_model)
    return encoders


def run(args, logger):
    k_batch_dict = make_batch_dict(args.k_batch_train, args.k_batch_test, args.k_batch_valid)
    n_batch_dict = make_batch_dict(args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    dataset = Hierarchical_Task(task_func_list, (k_batch_dict, n_batch_dict))
    
    base_model      = get_base_model(args)
    encoder_models  = get_encoder_model(args.encoders, args)
    model           = make_hierarhical_model(base_model, args.n_contexts, args.n_iters[:-1], args.lrs[:-1], encoder_models)

    # train(model, dataset.get('train'), args.n_iters[-1], args.lrs[-1], logger)     # Train
    debug(model, dataset.get('train'), args.n_iters[-1], args.lrs[-1], logger)     # Debug with finite diff
    
    test_loss = 0
    for minibatch_test in iter(dataset.get('test')):
        test_loss += model.evaluate(minibatch_test)

    return test_loss, logger

### Functions for finite difference method for calculating gradient

def check_grad(fnc, weights, eps = 1e-6, analytic_grad = False):
    grad_num = torch.zeros_like(weights)

    for i in range(weights.shape[0]):
        in_ = weights.clone().detach();  in_.requires_grad = True;       in_[i] += eps;             loss1 = fnc(in_);    
        in_ = weights.clone().detach(); in_.requires_grad = True;       in_[i] -= eps;             loss2 = fnc(in_)
        grad_num[i] = (loss1 - loss2)/2/eps
    if analytic_grad:
        in_ = weights.clone().detach()
        in_.requires_grad = True
        assert(in_.requires_grad)
        loss = fnc(weights.clone().detach());         loss.backward(); 
        return grad_num, in_.grad
    else:
        return grad_num


def eval_model_weight(model, minibatch, weights):
    # print('eval model', weights)
    # print(list(model.parameters()))
    # model.linear.weight = torch.nn.Parameter(weight)         #     model.linear.weight.data = weight
    torch.nn.utils.vector_to_parameters(weights, model.parameters())

    loss = model.evaluate(minibatch)
    return loss


def debug(model, dl, epochs, lr, logger):
    
    optim = Adam(model.parameters(), lr)
    # optim = torch.optim.SGD(model.parameters(), lr)

    # Getting all parameters
    model.double()

    for epoch in range(epochs):
        for minibatch in iter(dl):
            loss = model.evaluate(minibatch)
            
            optim.zero_grad()
            loss.backward()

            
            e = partial(eval_model_weight, model, minibatch)
            finite_grad = check_grad(e, torch.nn.utils.parameters_to_vector(model.parameters()), eps=1e-8, analytic_grad=False)

            t = []
            for p in model.parameters():
                t.append(p.grad.view(-1))
            analy_grad = torch.cat(t)

            # IPython.embed()
            print('grad error: ', (finite_grad - analy_grad).norm().detach().numpy())
            optim.step()

            # ------------ logging ------------
            logger.update(epoch, loss.detach().cpu().numpy())
            # vis_pca(higher_contexts, task_family, iteration, args)      ## Visualize result

def train(model, dl, epochs, lr, logger):
    
    optim = Adam(model.parameters(), lr)
    # optim = torch.optim.SGD(model.parameters(), lr)

    for epoch in range(epochs):
        for minibatch in iter(dl):
            loss = model.evaluate(minibatch)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            # ------------ logging ------------
            logger.update(epoch, loss.detach().cpu().numpy())
            # vis_pca(higher_contexts, task_family, iteration, args)      ## Visualize result


##############################################################################
#  Task Hierarchy
#  A 'task' has built-in sample() method, which returns a 'list of subtasks', and so on..
# 
#                                              super-duper-task (base_task)        f(., ., task_idx=None)  
# lv 2: task = super-duper-task,   subtasks  = super-tasks                        [f(., ., task_idx=None)]
# lv 1: task = super-task,         subtasks  = tasks (functions)                  [f(., ., task_idx)]
# lv 0: task = task (function),    subtasks  = data-points (inputs, targets)      [x, y= f(x, task_idx)]

def make_batch_dict(n_trains, n_tests, n_valids):
    return [{'train': n_train, 'test': n_test, 'valid': n_valid} for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids)]


class Hierarchical_Task():
    def __init__(self, task, batch_dict, inputs_gen=None):
        k_batch_dict, n_batch_dict = batch_dict
        self.task = task
        self.level = len(k_batch_dict) - 1
        self.k_batch = k_batch_dict[-1]
        self.n_batch = n_batch_dict[-1]
        self.batch_dict_next = (k_batch_dict[:-1], n_batch_dict[:-1])
        self.data = self.run_pre_sample()
        self.inputs_gen = inputs_gen

    def run_pre_sample(self):
        return {'train': self.pre_sample(self.k_batch['train'], self.n_batch['train'], sample_type='train'), 
                'test':  self.pre_sample(self.k_batch['test'], self.n_batch['test'], sample_type='test')}

    def high_level_presampling(self, K_batch, sample_type):
        if isinstance(self.task, list):
            assert K_batch <= len(self.task)
            tasks = random.sample(self.task, K_batch)
        else:
            tasks = [self.task(sample_type) for _ in range(K_batch)]
        return tasks

    # K_batch: total # of samples
    # N_batch: mini batch # of samples
    def pre_sample(self, K_batch, N_batch, sample_type):
        if self.level == 0:
            input_gen, target_gen = self.task
            input_data = input_gen(K_batch)
            target_data = target_gen(input_data)
            # IPython.embed()

            dataset = Level0_Dataset(x=input_data.double(), y=target_data.double())

            # DataLoader returns tensors
            return DataLoader(dataset, batch_size=N_batch, shuffle=True)

        else:
            tasks = self.high_level_presampling(K_batch, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]
            subtask_dataset = HighLevel_Dataset(x=subtask_list)
            
            # HighLevel_Dataloader returns a mini-batch of Hiearchical Tasks[
            return HighLevel_DataLoader(subtask_dataset, batch_size=N_batch)
    
    def get(self, sample_type):
        return self.data[sample_type]
         

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
        self.ctx = torch.zeros(1, self.n_context, requires_grad=True).double().to(self.device)

    def evaluate(self, minibatch, ctx_high = []):
        # assert self.level == tasks[0].level                                               # checking if the model level matches with task level
        loss = 0
        for task in minibatch:
            self.adaptation(task.get('train'), ctx_high)                                # adapt self.ctx  given high-level ctx 
            
            for minibatch_test in iter(task.get('test')):
                loss += self.submodel.evaluate(minibatch_test, ctx_high + [self.ctx])     # going 1 level down

        return loss / float(len(minibatch))

    def optimize(self, dl, ctx_high):                            # optimize parameter for a given 'task'
        self.reset_ctx()
        # optim = manual_optim([self.ctx], self.lr)                   # manual optim.SGD.  check for memory leak

        cur_iter = 0
        while True:        
            for minibatch in iter(dl):
                if cur_iter >= self.n_iter:
                    return False

                loss = self.submodel.evaluate(minibatch, ctx_high + [self.ctx])  
                self.ctx += torch.autograd.grad(loss, self.ctx, create_graph=True)[0]                 # grad = torch.autograd.grad(loss, model.ctx[level], create_graph=True)[0]                 # create_graph= not args.first_order)[0]

                # optim.zero_grad()
                # optim.backward(loss)
                # # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                # optim.step()             #  check for memory leak                                                         # model.ctx[level] = model.ctx[level] - args.lr[level] * grad            # if memory_leak:
                cur_iter += 1



    def forward(self, input, ctx_high = []):                                        # assuming ctx is optimized over training tasks already
        return self.submodel(input, ctx_high + [self.ctx])            

