import time
import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim import Adam
from model.models_huh import get_model_type
from task.mixture2 import task_func_list
from utils import manual_optim

torch.set_num_threads(3)


def get_base_model(args):
    MODEL_TYPE = get_model_type(args.model_type)
    model = MODEL_TYPE(n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
    return model


def get_encoder_model(encoder_types):
    encoders = []
    for encoder_type in encoder_types:
        if encoder_type is None:
            encoders.append(None)
        else:
            raise NotImplementedError()
    return encoders


def run(args, logger):
    k_batch_dict = make_batch_dict(args.k_batch_train, args.k_batch_test, args.k_batch_valid)
    n_batch_dict = make_batch_dict(args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    dataset = Hierarchical_Task(task_func_list, (k_batch_dict, n_batch_dict))

    base_model = get_base_model(args)
    encoder_models = get_encoder_model(args.encoders)
    model = make_hierarhical_model(base_model, args.n_contexts, args.n_iters[:-1], args.lrs[:-1], encoder_models)

    train(model, dataset, args.n_iters[-1], args.lrs[-1], logger)
    test_loss = model.evaluate(dataset.sample('test'))
    return test_loss, logger


def train(model, task, n_iter, lr, logger):
    optim = Adam(model.parameters(), lr)

    for iter in range(n_iter): 
        loss = model.evaluate(task.sample('train'))
        optim.zero_grad()
        loss.backward()
        optim.step()

        logger.update(iter, loss.detach().cpu().numpy())


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
    def __init__(self, task, batch_dict):
        k_batch_dict, n_batch_dict = batch_dict
        self.task = task
        self.level = len(k_batch_dict) - 1
        self.k_batch = k_batch_dict[-1]
        self.n_batch = n_batch_dict[-1]
        self.batch_dict_next = (k_batch_dict[:-1], n_batch_dict[:-1])
        self.data = self.run_pre_sample()

    def run_pre_sample(self):
        return {
            'train': self.pre_sample(self.k_batch['train']), 
            'test': self.pre_sample(self.k_batch['test'])}

    def pre_sample(self, K_batch):
        if self.level == 0:
            n_input = 1
            input_data = torch.randn(K_batch, n_input)
            target_data = self.task(input_data)
            return [(input_data[i, :], target_data[i, :]) for i in range(K_batch)]
        else:
            if isinstance(self.task, list):
                assert K_batch <= len(self.task)
                tasks = random.sample(self.task, K_batch)
            else:
                tasks = [self.task() for _ in range(K_batch)]
            return [self.__class__(task, self.batch_dict_next) for task in tasks]

    def sample(self, sample_type):
        batch = self.n_batch[sample_type]
        dataset = random.sample(self.data[sample_type], batch)
        if self.level == 0:
            input_data = torch.cat([data[0] for data in dataset], dim=0).view(batch, -1)
            target_data = torch.cat([data[1] for data in dataset], dim=0).view(batch, -1)
            return [input_data, target_data]
        else: 
            return dataset


##############################################################################
#  Model Hierarchy
def make_hierarhical_model(model, n_contexts, n_iters, lrs, encoders):
    for level, (n_context, n_iter, lr, encoder) in enumerate(zip(n_contexts, n_iters, lrs, encoders)):
        model = Hierarchical_Model(model, level, n_context, n_iter, lr, encoder)
    return model


class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, decoder_model, level, n_context, n_iter, lr, encoder_model=None): 
        super().__init__()
        assert hasattr(decoder_model, 'evaluate')    # submodel has evaluate() method built-in
        self.submodel = decoder_model 
        self.level = level                  # could be useful for debugging/experimenting
        self.n_context = n_context
        self.n_iter = n_iter
        self.lr = lr
        self.device = decoder_model.device
        self.adaptation = self.optimize if encoder_model is None else encoder_model 

    def evaluate(self, tasks, ctx_high=[]):
        assert self.level == tasks[0].level

        # At super-task level, apply no threading
        loss = 0. 
        if len(tasks) == 2:
            for task in tasks:
                ctx = self.adaptation(task.sample('train'), ctx_high)
                loss += self.submodel.evaluate(task.sample('test'), ctx_high + [ctx])

        # At lower level, apply threading
        else:
            processes, context_dict = [], mp.Manager().dict()
            for i_task, task in enumerate(tasks):
                p = mp.Process(
                    target=self.adaptation,
                    args=(task.sample('train'), ctx_high, context_dict, i_task))
                p.start()
                processes.append(p)
                time.sleep(0.01)

            for p in processes:
                time.sleep(0.01)
                p.join()

            for i_task, ctx in context_dict.items():
                task = tasks[i_task]
                ctx = torch.from_numpy(ctx)
                loss += self.submodel.evaluate(task.sample('test'), ctx_high + [ctx])

        return loss / float(len(tasks))  

    def optimize(self, tasks, ctx_high, context_dict=None, rank=None):
        ctx = torch.zeros(1, self.n_context, requires_grad=True).to(self.device)
        optim = manual_optim([ctx], self.lr)

        for _ in range(self.n_iter): 
            loss = self.submodel.evaluate(tasks, ctx_high + [ctx])  
            optim.zero_grad()
            optim.backward(loss)
            optim.step()

        if context_dict is None:
            return ctx.detach()
        else:
            context_dict[rank] = ctx.detach().cpu().numpy()
