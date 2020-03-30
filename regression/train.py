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
from models import CaviaModel, Model_Active, Encoder_Decoder, Onehot_Encoder, Encoder_Variational
from logger import Logger
import IPython


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


def get_task_family(args):
    task_family = {}
    if args.task == 'sine':
        task_family['train'] = tasks_sine.RegressionTasksSinusoidal()
        task_family['valid'] = tasks_sine.RegressionTasksSinusoidal()
        task_family['test'] = tasks_sine.RegressionTasksSinusoidal()
    elif args.task == 'celeba':
        task_family['train'] = tasks_celebA.CelebADataset('train', device=args.device, celeba_dir=args.celeba_dir)
        task_family['valid'] = tasks_celebA.CelebADataset('valid', device=args.device, celeba_dir=args.celeba_dir)
        task_family['test'] = tasks_celebA.CelebADataset('test', device=args.device, celeba_dir=args.celeba_dir)
    else:
        raise NotImplementedError
    return task_family


def get_model_decoder(args, task_family_train):
    n_arch = args.architecture

    # assert  n_arch[0] == task_family_train.num_inputs 
    # assert  n_arch[-1] == task_family_train.num_outputs 

    n_context = args.num_context_params

    if args.model_type == 'cavia':
        MODEL = CaviaModel
        n_arch[0] +=  n_context
    elif args.model_type == 'active':
        MODEL = Model_Active
        

    model_decoder = MODEL(n_arch=n_arch, n_context=n_context, device=args.device).to(args.device)
    return model_decoder

def get_model_encoder(args, task_family_train):
    n_context = args.num_context_params
    n_task = args.tasks_per_metaupdate
    n_total_tasks = args.total_num_tasks

    if args.encoder == '1hot':
        model_encoder = Onehot_Encoder(n_context, n_total_tasks)
    elif args.encoder == 'vae':
        model_encoder = Encoder_Variational(task_family_train.num_inputs, n_context, 64)

    
    return model_encoder 


def update_logger(logger, path, args, model, eval_fnc, task_family, i_iter, start_time):

    def logger_helper(logger, args, model, eval_fnc, task_family, task_type):
        loss_mean, loss_conf = eval_fnc(args, copy.deepcopy(model), task_family=task_family[task_type], num_updates=args.num_inner_updates)
        getattr(logger, task_type+'_loss').append(loss_mean)
        getattr(logger, task_type+'_conf').append(loss_conf)
        return logger

    # evaluate on training set
    logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='train')
    logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='test')
    logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='valid')

    # save logging results
    utils.save_obj(logger, path)
    # save best model
    if logger.valid_loss[-1] == np.min(logger.valid_loss):
        logger.update_best_model(model)
        print('saving best model at iter', i_iter)
    # visualise results	
    if args.task == 'celeba':
        task_family['train'].visualise(task_family['train'], task_family['test'], copy.deepcopy(logger.best_valid_model), args, i_iter)
    # print current results
    logger.print_info(i_iter, start_time)
    start_time = time.time()
    return logger, start_time


def eval_model(model, inputs, target_fnc, context=None):
    outputs = model(inputs, context)
    targets = target_fnc(inputs)
    return F.mse_loss(outputs, targets)

def eval_model_1hot(model, inputs, target_fnc, labels_1hot):
    outputs = model(inputs, labels_1hot)
    targets = target_fnc(inputs)
    return F.mse_loss(outputs, targets)

def inner_update_step(args, model, context, train_inputs, target_fnc, eval_cavia = False):
    # update context on current task 
    task_loss = eval_model(model, context, train_inputs, target_fnc)
    task_gradients = torch.autograd.grad(task_loss, context, create_graph=not args.first_order)[0]
    # update context params (this will set up the computation graph correctly)

    if eval_cavia and args.first_order:  # ?? is this necessary??
        task_gradients = task_gradients.detach()

    context = context - args.lr_inner * task_gradients
    return context


def get_meta_gradient(args, model, task_family, target_fnc):

    train_inputs = task_family['train'].sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
    test_inputs = task_family['test'].sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)

    context = model.reset_context()
    # inner-loop update of context
    for _ in range(args.num_inner_updates):
        context = inner_update_step(args, model, context, train_inputs, target_fnc)

    # ------------ compute meta-gradient on test loss of current task ------------
    loss_meta = eval_model(model, context, test_inputs, target_fnc)
    grad_meta = torch.autograd.grad(loss_meta, model.parameters())

    return grad_meta


def meta_backward(args, model, task_family, target_functions):
    meta_gradient = [0 for _ in range(len(model.state_dict()))]     

    # --- compute meta gradient ---
    for t in range(args.tasks_per_metaupdate):
        target_fnc = target_functions[t]
        grad_meta = get_meta_gradient(args, model, task_family, target_fnc)

        # clip the gradient and accumulate
        for i in range(len(grad_meta)):
            meta_gradient[i] += grad_meta[i].detach().clamp_(-10, 10)

    for i, param in enumerate(model.parameters()):
        param.grad = meta_gradient[i] / args.tasks_per_metaupdate

def meta_backward_1hot(args, model, inner_optimizer, outer_optimizer, task_family, target_functions, labels_1hot):
    # --- compute meta gradient ---
    for t in range(args.tasks_per_metaupdate):
        
        train_inputs = task_family['train'].sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
        test_inputs = task_family['test'].sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)
        
        # Encoder / inner update
        update_step_1hot(model, inner_optimizer, train_inputs, target_functions[t], labels_1hot[t])
        # Decoder / outer update -- test loss computed with test inputs
        train_loss = update_step_1hot(model, outer_optimizer, test_inputs, target_functions[t], labels_1hot[t])
        #print('in meta bcakward loss {}'.format(train_loss.item()))


def update_step_1hot(model, optimizer, train_inputs, target_fnc, label_1hot):
    # Update given optimizer (whether inner or outer)
    optimizer.zero_grad()
    loss = eval_model_1hot(model, train_inputs, target_fnc, label_1hot)
    loss.backward()
    optimizer.step()

    return loss


# Not sure if necessary -- change
def get_inputs_outputs_1hot(args, task_family):
    target_functions = task_family.sample_tasks(args.tasks_per_metaupdate)
    tasks_onehot = task_family.sample_tasks_onehot(num_tasks=args.tasks_per_metaupdate, batch_size=args.k_meta_train)
    inputs = task_family.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

    return inputs, target_functions, tasks_onehot


#########################################

def run(args, log_interval=5000, rerun=False):

    path = initial_setting(args, rerun)
    
    # initialise 
    task_family = get_task_family(args)
    model = get_model_decoder(args, task_family['train'])
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)
    logger = Logger(model)

    start_time = time.time()
    # --- main training loop ---

    for i_iter in range(args.n_iter):

        # sample tasks
        target_functions = task_family['train'].sample_tasks(args.tasks_per_metaupdate)
        meta_backward(args, model, task_family, target_functions)
        meta_optimiser.step()

        # ------------ logging ------------
        if i_iter % log_interval == 0:
            logger, start_time = update_logger(logger, path, args, model, eval_cavia, task_family, i_iter, start_time)

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
        target_fnc = task_family.sample_task()
        curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)

        context = model.reset_context()
        # ------------ update context on current task ------------
        for _ in range(1, num_updates + 1):
            context = inner_update_step(args, model, context, curr_inputs, target_fnc, eval_cavia=True)

        # compute true loss on entire input range
        model.eval()
        losses.append(eval_model(model, context, input_range, target_fnc).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)

def run_no_inner(args, log_interval=5000, rerun=False):
    path = initial_setting(args, rerun)
    task_family = get_task_family(args)

    # Create model
    model_encoder = get_model_encoder(args, task_family['train'])
    model_decoder = get_model_decoder(args, task_family['train'])
    
    model = Encoder_Decoder(model_encoder, model_decoder)

    # initialise loggers
    logger = Logger(model)

    # intitialise optimisers
    outer_optimizer = optim.Adam(model.decoder.parameters(), args.lr_meta)
    inner_optimizer = optim.SGD(model.encoder.parameters(), args.lr_inner, weight_decay=args.weight_decay)

    start_time = time.time()

    # --- main training loop ---

    for i_iter in range(args.n_iter):
        # Sample new mini-batch of tasks from pre-sampled batch of tasks
        target_functions, train_labels_onehot = task_family['train'].sample_tasks_1hot(total_num_tasks=args.total_num_tasks, batch_num_tasks=args.tasks_per_metaupdate, batch_size=args.k_meta_train)
        meta_backward_1hot(args, model, inner_optimizer, outer_optimizer, task_family, target_functions, train_labels_onehot)

        if i_iter % log_interval == 0:
            logger, start_time = update_logger(logger, path, args, copy.deepcopy(model), eval_1hot, task_family, i_iter, start_time)        

    return logger

def eval_1hot(args, model, task_family, num_updates, n_tasks=25, return_gradnorm=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)
    
    # logging
    losses = []
    gradnorms = []

    # Optimizer for encoder (context)
    inner_optimizer = optim.SGD(model.encoder.parameters(), args.lr_inner)

    # Sample tasks
    target_functions, valid_labels_1hot = task_family.sample_tasks_1hot(total_num_tasks=n_tasks*4, batch_num_tasks=n_tasks, batch_size=args.k_shot_eval)
    
    # Reinitialize one hot encoder for evaluation mode
    model.encoder.reinit(args.num_context_params, task_family.total_num_tasks)

    # --- inner loop ---

    for t in range(n_tasks):
        # Get 1hot labels of total input range: This need to be fixed
        input_range_1hot = task_family.create_input_range_1hot_labels(batch_size=input_range.shape[0], cur_label=valid_labels_1hot[t, 0, :])
        
        valid_inputs = task_family.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
        
        # ------------ update on current task ------------

        for _ in range(1, num_updates + 1):
            # forward pass
            update_step_1hot(model, inner_optimizer, valid_inputs, target_functions[t], valid_labels_1hot[t])

        # ------------ logging ------------

        # compute true loss on entire input range
        model.eval()
        losses.append(eval_model_1hot(model, input_range, target_functions[t], input_range_1hot).item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)

