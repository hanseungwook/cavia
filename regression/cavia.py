"""
Regression experiment using CAVIA
"""
import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
import tasks_sine, tasks_celebA
from models import CaviaModel, Model_Active, Encoder_Decoder, Onehot_Encoder
from logger import Logger


############ should rename the file... it's not just cavia anymore ##############
############ Check for any bugs ###############


def initial_setting(args, rerun):
    assert not args.maml

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)

    if os.path.exists(path + '.pkl') and not rerun:
        return utils.load_obj(path)

    utils.set_seed(args.seed)
    
    return path



def get_task_family(task):
    task_family = {}
    if task == 'sine':
        task_family['train'] = tasks_sine.RegressionTasksSinusoidal()
        task_family['valid'] = tasks_sine.RegressionTasksSinusoidal()
        task_family['test'] = tasks_sine.RegressionTasksSinusoidal()
    elif task == 'celeba':
        task_family['train'] = tasks_celebA.CelebADataset('train', device=args.device)
        task_family['valid'] = tasks_celebA.CelebADataset('valid', device=args.device)
        task_family['test'] = tasks_celebA.CelebADataset('test', device=args.device)
    else:
        raise NotImplementedError
    return task_family


def get_model_decoder(args, task_family_train):
    n_arch = args.architecture

    assert  n_arch[0] == task_family_train.num_inputs 
    assert  n_arch[-1] == task_family_train.num_outputs 

    n_context = args.num_context_params

    if args.model_type == 'CAVIA':
        MODEL = CaviaModel
        n_arch[0] +=  n_context
    elif args.model_type == 'ACTIVE':
        MODEL = Model_Active
        

    model_decoder = MODEL(n_arch=n_arch,  n_context=n_context,  device=args.device).to(args.device)
    return model_decoder


def update_logger(logger, path, args, model, eval_fnc, task_family, i_iter, start_time):

    def logger_helper(logger, args, model, eval_fnc, task_family, task_type):
        loss_mean, loss_conf = eval_fnc(args, copy.deepcopy(model), task_family=task_family[task_type],  num_updates=args.num_inner_updates)
        getattr(logger, task_type+'_loss').append(loss_mean)
        getattr(logger, task_type+'_conf').append(loss_conf)
        return logger

    # evaluate on training set
    logger = logger_helper(logger, args, model,  eval_fnc, task_family, task_type = 'train')
    logger = logger_helper(logger, args, model,  eval_fnc, task_family, task_type = 'test')
    logger = logger_helper(logger, args, model,  eval_fnc, task_family, task_type = 'valid')

    # save logging results
    utils.save_obj(logger, path)
    # save best model
    if logger.valid_loss[-1] == np.min(logger.valid_loss):
        print('saving best model at iter', i_iter)
        logger.best_valid_model = copy.deepcopy(model)
    # visualise results	
    if args.task == 'celeba':
        task_family['train'].visualise(task_family['train'], task_family['test'], copy.deepcopy(logger.best_valid_model), args, i_iter)
    # print current results
    logger.print_info(i_iter, start_time)
    start_time = time.time()
    return logger, start_time




def eval_model(model, context, inputs, target_fnc):
    outputs = model(inputs, context)
    targets = target_fnc(inputs)
    return F.mse_loss(outputs, targets)



def inner_loop_step(args, model, task_family, target_fnc, meta_gradient):

    # get data for current task
    train_inputs = task_family['train'].sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
    # reset private network weights
    context = model.reset_context()

    for _ in range(args.num_inner_updates):

        # ------------ update context_params on current task ------------
        # gradient wrt context params for current task
        task_loss = eval_model(model, context, train_inputs, target_fnc)
        task_gradients =  torch.autograd.grad(task_loss, context, create_graph=not args.first_order)[0]
        # update context params (this will set up the computation graph correctly)
        context = context - args.lr_inner * task_gradients

    # ------------ compute meta-gradient on test loss of current task ------------

    ### SHOULDn't THIS USE task_family['test'] instead?? ####
    test_inputs = task_family['train'].sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)
    loss_meta = eval_model (model, context, test_inputs, target_fnc)
    # compute gradient + save for current task
    task_grad = torch.autograd.grad(loss_meta, model.parameters())

    # clip the gradient
    for i in range(len(task_grad)):
        meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

    return meta_gradient




#########################################



def run(args, log_interval=5000, rerun=False):

    path = initial_setting(args, rerun)
    
    # initialise 
    task_family = get_task_family(args.task)
    model = get_model_decoder(args, task_family['train'])
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    start_time = time.time()
    # --- main training loop ---

    for i_iter in range(args.n_iter):

        # sample tasks
        target_functions = task_family['train'].sample_tasks(args.tasks_per_metaupdate)
        meta_gradient = [0 for _ in range(len(model.state_dict()))]     

        # --- inner loop ---
        for t in range(args.tasks_per_metaupdate):
            target_fnc = target_functions[t]
            meta_gradient = inner_loop_step(args, model, task_family, target_fnc, meta_gradient)

        # ------------ meta update ------------

        for i, param in enumerate(model.parameters()):
            param.grad = meta_gradient[i] / args.tasks_per_metaupdate

        # do update step on shared model
        meta_optimiser.step()

        # # reset context params
        # model.reset_context_params()

        # ------------ logging ------------
        if i_iter % log_interval == 0:
            logger, start_time = update_logger(logger, path, args, model, eval_cavia, task_family, i_iter, start_time)

    return logger

def run_no_inner(args, log_interval=5000, rerun=False):
    path = initial_setting(args, rerun)

    task_family = get_task_family(args.task)

    model_decoder = get_model_decoder(args, task_family['train'])
    model = Encoder_Decoder(model_decoder, n_context=args.num_context_params, n_task=1)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # intitialise optimisers
    outer_optimiser = optim.Adam(model.decoder.parameters(), args.lr_meta)
    inner_optimiser = optim.SGD(model.encoder.parameters(), args.lr_inner)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    start_time = time.time()

    # --- main training loop ---

    for i_iter in range(args.n_iter):

        # Sample tasks
        target_functions = task_family['train'].sample_tasks(args.tasks_per_metaupdate)
        train_tasks_onehot = task_family['train'].sample_tasks_onehot(num_tasks=1, batch_size=args.k_meta_train)
        train_inputs = task_family['train'].sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

        # Inner & outer loop being done alternatively
        for t in range(args.tasks_per_metaupdate):
            train_targets = target_functions[t](train_inputs)

            train_outputs = model(train_inputs, train_tasks_onehot)

            loss = F.mse_loss(train_outputs, train_targets)
        
            inner_optimiser.zero_grad()
            outer_optimiser.zero_grad()

            loss.backward()

            inner_optimiser.step()
            outer_optimiser.step()

        if i_iter % log_interval == 0:
            logger, start_time = update_logger(logger, path, args, model, eval_1hot, task_family, i_iter, start_time)        

    return logger

def eval_cavia(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function = task_family.sample_task()

        # reset context parameters
        context = model.reset_context()

        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
        curr_targets = target_function(curr_inputs)

        # ------------ update on current task ------------

        for _ in range(1, num_updates + 1):

            # forward pass
            curr_outputs = model(curr_inputs, context)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            task_gradients = \
                torch.autograd.grad(task_loss, context, create_graph=not args.first_order)[0]

            # update context params
            if args.first_order:
                context = context - args.lr_inner * task_gradients.detach()
            else:
                context = context - args.lr_inner * task_gradients

            # keep track of gradient norms
            gradnorms.append(task_gradients[0].norm().item())

        # ------------ logging ------------

        # compute true loss on entire input range
        model.eval()
        losses.append(F.mse_loss(model(input_range, context), target_function(input_range)).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)


def eval_1hot(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)
    input_range_1hot = task_family.sample_tasks_onehot(num_tasks=1, batch_size=input_range.shape[0])

    # logging
    losses = []
    gradnorms = []

    # Optimizer for encoder (context)
    inner_optimiser = optim.SGD(model.encoder.parameters(), args.lr_inner)

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function = task_family.sample_task()

        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
        curr_1hot = task_family.sample_tasks_onehot(num_tasks=1, batch_size=args.k_shot_eval)
        curr_targets = target_function(curr_inputs)

        # ------------ update on current task ------------

        for _ in range(1, num_updates + 1):

            # forward pass
            curr_outputs = model(curr_inputs, curr_1hot)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)
            task_loss.backward()

            # Calculating gradient norm
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # keep track of gradient norms
            gradnorms.append(total_norm)

            inner_optimiser.step()

        # ------------ logging ------------

        # compute true loss on entire input range
        model.eval()
        losses.append(F.mse_loss(model(input_range, input_range_1hot), target_function(input_range)).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)

