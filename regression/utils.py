import logging
# from tensorboardX import SummaryWriter

import os, sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA




import IPython
from pdb import set_trace

DEBUG_LEVELS = []  # [1] #[0]  #[2]


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

##################################
def move_to_device(input_tuple, device):
    if device in (None, 'cpu') :
        return input_tuple
    else:
        return [k.to(device) for k in input_tuple]
    
def check_nan(loss):
    if torch.isnan(loss):
        print("loss is nan")
        set_trace()

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
    for_iter = args_dict['for_iters'][level]
#     logger = args_dict['loggers'][level] 
    return lr, max_iter, for_iter, #logger

#################################################################################
# LOGGING
#################################################################################


def get_loggers(logger_maker, levels):
    def get_logger_list(log_type):
        logger_list = []
        for i in range(levels):
            no_print=(i<levels-1)
            logger_name = log_type+'_lv'+str(i)
            logger_list.append(logger_maker(additional_name=logger_name, no_print=no_print))
            if print_logger_name:
                print('logger_name=', logger_name, 'no_print=', no_print)
        return logger_list
    return get_logger_list(log_type='train'), get_logger_list(log_type='test')


class Logger():
    def __init__(self, args, additional_name='', no_print=False):
        self.log         = set_log(args, no_print)
        self.log_name    = args.log_name
        self.update_iter = args.log_interval
        self.tb_writer   = SummaryWriter('./logs/tb_{}_{}'.format(args.log_name, additional_name))
        self.no_print = no_print
        self.iter = 0
        
    def close():  # Huh: SummaryWriter must be closed at the end. Currently it never gets called. Problem!
        self.tb_writer()

    def log_loss(self, loss, level=2, num_adapt=None):
        # print(iter, self.update_iter)
        if not self.no_print and not (self.iter % self.update_iter):
            self.log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))

        if level < 2:
            self.tb_writer.add_scalar("Meta loss/Adapt{}".format(num_adapt), loss, self.iter)
        
        self.tb_writer.add_scalar("Meta loss/Total", loss, self.iter)
        
        self.iter += 1
    
    def log_ctx(self, task_name, ctx):
        # self.log[self.log_name].info('Logging context at iteration {}'.format(self.iter))
        self.tb_writer.add_histogram("Context {}".format(task_name), ctx, self.iter)

        # Log each context changing separately if size <= 5
        if ctx.size <= 5:
            for i in range(ctx.size):
                self.tb_writer.add_scalar("Context {}/{}".format(task_name, i), ctx[i], self.iter)


def set_log(args, no_print):
    log = {}
    set_logger(logger_name=args.log_name,   log_file=r'{0}{1}'.format("./logs/",  args.log_name))  
    log[args.log_name] = logging.getLogger(args.log_name)

    # Only print if logger is set to print
    # if not no_print:
    #     for arg, value in sorted(vars(args).items()):
    #         log[args.log_name].info("%s: %r", arg, value)

    return log


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)

    if not getattr(log, 'handler_set', None):
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        log.setLevel(level)
        log.addHandler(fileHandler)
        log.addHandler(streamHandler)



##########################

def print_args(args):
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value), file=sys.stderr)



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


#################################################################################
# VISUALIZATION
#################################################################################
def get_vis_fn(tasks):
    for task in tasks:
        if task == 'cifar10' or task == 'mnist' or task == 'fmnist' or task == 'mnist_fmnist' or task == 'mnist_fmnist_3level':
            return vis_save_img_recon
        else:
            raise NotImplementedError()

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

def vis_save_img_recon(outputs, save_dir, itr):
    img_pred = outputs.view(28, 28, 1).detach().cpu().numpy()

    # Forcing all predictions beyond image value range into (0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    img_pred = np.round(img_pred * 255.0).astype(np.uint8)
    img_pred = transforms.ToPILImage(mode='L')(img_pred)
    img_pred.save(os.path.join(save_dir, 'recon_img_itr{}.png'.format(itr)))
