import logging
from tensorboardX import SummaryWriter

import os
import random
import torch
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import higher

from finite_diff import debug_lower, debug_top
# from hierarchical import optimize
from task.mixture2 import get_img_full_input
from task import mixture2




import IPython
from pdb import set_trace

DEBUG_LEVELs = []  # [1] #[0]  #[2]


#################################################################################
# LOGGING
#################################################################################

class Logger():
    def __init__(self, args, additional_name='', no_print=False):
        self.log         = set_log(args)
        self.log_name    = args.log_name
        self.update_iter = args.log_interval
        self.tb_writer   = SummaryWriter('./logs/tb_{}_{}'.format(args.log_name, additional_name))
        self.no_print = no_print

    def update(self, iter, loss):
        if not (iter % self.update_iter):
            # print(iter, self.update_iter)
            if not self.no_print:
                self.log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format( iter, loss))

            self.tb_writer.add_scalar("Meta loss:", loss, iter)


def set_log(args):
    log = {}
    set_logger(logger_name=args.log_name,   log_file=r'{0}{1}'.format("./logs/",  args.log_name))  
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



##########################

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
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

###############################
def get_args(args_dict, level):
    # return (arg[name][level] for arg, name in args_dict.items())
    lr = args_dict['lrs'][level] 
    max_iter = args_dict['max_iters'][level] 
    logger = args_dict['loggers'][level] 
    return lr, max_iter, logger


#################################################################################
# VISUALIZATION
#################################################################################
def get_vis_fn(tasks):
    for task in tasks:
        if task == 'cifar10':
            return vis_img_recon

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

def vis_img_recon(model, task):    
    task = task[0]
    max_level = task.level
    level = max_level
    
    # From higher levels, recurse all the way down to level 0
    while level > 0:
        print(task)
        train_loader = task.loader['train']
        task = next(iter(train_loader))[0]
        level -= 1
    
    # Input and target generator functions for a specific task from train (level 0)
    input_gen, target_gen = task.task
    img_inputs, img_targets = next(iter(task.loader['train']))

    # Inner loop optimizations
    level = max_level - 1
    reset = True

    # 0-level optimization is fine, but for 1-level optimization, usually we give a batch of images (coordinates and targets) that's in the same class,
    # but here, we only give 1 image from 1 class => is this problematic?
    while level >= 0:
        lr, max_iter, logger = get_args(model.args_dict, level)
        param_all = model.decoder_model.parameters_all

        ## Initialize param & optim
        if reset:
            param_all[level] = torch.zeros_like(param_all[level], requires_grad= True)   # Reset
            optim = SGD([param_all[level]], lr=lr)
            optim = higher.get_diff_optim(optim, [param_all[level]]) #, device=x.device) # differentiable optim for inner-loop:
        
        cur_iter = 0
        while True:
            print('cur iter / max iter', cur_iter, max_iter)
            if cur_iter >= max_iter:    
                break

            loss, _ = model.decoder_model((img_inputs, img_targets))

            ## loss.backward() & optim.step()
            if reset:
                new_param, = optim.step(loss, params=[param_all[level]])   # syntax for diff_optim
                param_all[level] = new_param

            cur_iter += 1   
        
        level -= 1
    
    # Get full input range of image
    img_full_inputs = get_img_full_input()

    # Get real and predicted image
    img_real = target_gen(img_full_inputs).view(mixture2.img_size).numpy()
    img_pred = model.decoder_model((img_full_inputs, )).view(mixture2.img_size).detach().numpy()

    # Forcing all predictions beyond image value range into (0, 1)
    img_pred = np.clip(img_pred, 0, 1)

    # Plotting real and predicted images
    fig = plt.figure(figsize=(16, 16))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_real)

    fig.add_subplot(1, 2, 2)
    plt.imshow(img_pred)

    plt.show()

