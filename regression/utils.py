import logging
from tensorboardX import SummaryWriter

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.log = set_log(args)
        self.tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))

    def update(self, iter, loss):
        if not (iter % self.args.log_interval):
            self.log[self.args.log_name].info("At iteration {}, meta-loss: {:.3f}".format(iter, loss))
            self.tb_writer.add_scalar("Meta loss:", loss, iter)


def set_log(args):
    log = {}
    set_logger(logger_name=args.log_name, log_file=r'{0}{1}'.format("./logs/", args.log_name))  
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    return log


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_seed(seed, cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # NOTE Below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


#####################################
# Manual optim :  replace optim.SGD  due to memory leak problem

class manual_optim():
    def __init__(self, param_list, lr, grad_clip = None, first_order = False):
        self.param_list = param_list
        self.lr = lr
        self.grad_clip = grad_clip

    def zero_grad(self):
        for par in self.param_list:
            # assert par.requires_grad #(par.requires_grad == True)
            par.grad = torch.zeros_like(par.data)               # par.grad = torch.zeros(par.data.shape, device=par.device)   

    def backward(self, loss):
        grad_list = torch.autograd.grad(loss, self.param_list, create_graph=True)
        for par, grad in zip(self.param_list, grad_list):
            par.grad = grad

    def step(self):
        # Gradient clipping
        # if self.grad_clip is not None:
        #     torch.nn.utils.clip_grad_norm_(self.param_list, self.grad_clip)

        for par in self.param_list:
            # par.data = par.data - self.lr * par.grad  # does not construct computational graph. 
            # par.data -= self.lr * par.grad            # does not construct computational graph. 
            # par = par - self.lr * par.grad            # GOOOD !!!
            par -= par.grad * self.lr                 # # also good


###############################
def send_to(input, device, DOUBLE_precision):
    if DOUBLE_precision:
        return input.double().to(device)
    else:
        return input.double().float().to(device)    # somehow needed for manual_optim to work.... otherwise get leaf node error. 


#################################################################################
# VISUALIZATION
#################################################################################
def vis_pca(higher_contexts, task_family, iteration, args):
    pca = PCA(n_components=2)
    higher_contexts = torch.stack(higher_contexts).detach().cpu().numpy()
    higher_contexts_pca = pca.fit_transform(higher_contexts)

    # TODO Consider same PCA dimension
    # TODO Consider also plotting lower context variable
    for i_super_task, super_task in enumerate(task_family["train"].super_tasks):
        x, y = higher_contexts_pca[i_super_task, :] 
        plt.scatter(x, y, label=super_task)
        print(x, y)
    plt.legend()
    plt.title("PCA_iteration" + str(iteration))
    plt.xlim([-1., 1.])
    plt.ylim([-1., 1])
    plt.savefig("logs/n_inner" + str(args.n_inner) + "/pca_iteration" + str(iteration).zfill(3) + ".png")
    plt.close()


def vis_prediction(model, lower_context, higher_context, inputs, task_function, super_task, iteration, args):
    # Create directories
    if not os.path.exists("./logs/n_inner" + str(args.n_inner)):
        os.makedirs("./logs/n_inner" + str(args.n_inner))

    outputs = model(inputs, lower_context, higher_context).detach().cpu().numpy()
    targets = task_function(inputs).detach().cpu().numpy()

    plt.figure()
    plt.scatter(inputs, outputs, label="pred")
    plt.scatter(inputs, targets, label="gt")
    plt.legend()
    plt.title(super_task + "_iteration" + str(iteration))

    plt.savefig("logs/n_inner" + str(args.n_inner) + "/iteration" + str(iteration).zfill(3) + "_" + super_task + ".png")
    plt.close()
