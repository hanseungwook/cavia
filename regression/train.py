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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import utils
from tasks import sine, celeba
from models import CaviaModel, Model_Active, Encoder_Decoder, Encoder_Decoder_VAE, Onehot_Encoder, Encoder_Variational
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

def create_targets(inputs, target_functions, n_tasks):
    bs = int(inputs.shape[0] / n_tasks)
    return torch.cat([target_functions[t](inputs[bs*t:bs*(t+1)]) for t in range(n_tasks)])

def create_1hot_labels(labels_1hot, n_tasks):
    return torch.cat([labels_1hot[t] for t in range(n_tasks)]).reshape(-1, labels_1hot.shape[2])

def viz_pred(model, inputs, targets, labels, task_type=None):
    preds = model(inputs, labels)
    if len(preds) == 3:
        preds = preds[0]
    
    plt.scatter(inputs.detach().numpy(), targets.detach().numpy(), c='green', label='Target', s=1)
    plt.scatter(inputs.detach().numpy(), preds.detach().numpy(), c='blue', label='Pred', s=1)
    plt.legend()

    if task_type:
        plt.title(task_type)
    plt.show()

def get_task_family(args):
    task_family = {}
    if args.task == 'sine':
        task_family['train'] = sine.RegressionTasksSinusoidal()
        task_family['valid'] = sine.RegressionTasksSinusoidal()
        task_family['test'] = sine.RegressionTasksSinusoidal()
    elif args.task == 'celeba':
        task_family['train'] = celeba.CelebADataset('train', device=args.device, celeba_dir=args.celeba_dir)
        task_family['valid'] = celeba.CelebADataset('valid', device=args.device, celeba_dir=args.celeba_dir)
        task_family['test'] = celeba.CelebADataset('test', device=args.device, celeba_dir=args.celeba_dir)
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
        num_total_inputs = task_family_train.num_inputs + task_family_train.num_outputs
        model_encoder = Encoder_Variational(num_total_inputs, n_context, 64)

    return model_encoder


def update_logger(logger, path, args, model, eval_fnc, task_family, i_iter, start_time):

    def logger_helper(logger, args, model, eval_fnc, task_family, task_type):
        loss_mean, loss_conf = eval_fnc(args, copy.deepcopy(model), task_family=task_family[task_type], num_updates=args.num_inner_updates, task_type=task_type)
        getattr(logger, task_type+'_loss').append(loss_mean)
        getattr(logger, task_type+'_conf').append(loss_conf)
        return logger

    # evaluate on training set
    logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='train')
    # logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='valid')
    # logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='test')

    # save logging results
    utils.save_obj(logger, path)
    # save best model
    if logger.valid_loss[-1] == np.min(logger.valid_loss):
        logger.update_best_model(model)
        print('saving best model at iter', i_iter)
    # visualise results	
    if args.task == 'celeba':
        task_family['train'].visualise(task_family['train'], task_family['test'], copy.deepcopy(logger.best_valid_model), args, i_iter)

    logger.print_info(i_iter, start_time)
    # start_time = time.time()
    return logger, start_time


def eval_model(model, inputs, target_fnc, context=None):
    outputs = model(inputs, context)
    targets = target_fnc(inputs)
    
    return F.mse_loss(outputs, targets)

def eval_model_1hot(model, inputs, targets, labels_1hot):
    outputs = model(inputs, labels_1hot)
    
    return F.mse_loss(outputs, targets)


def inner_update(args, model, task_family, target_fnc):
    context = model.reset_context()
    inner_optim = optim.SGD([context], args.lr_inner) #     optim_inner.zero_grad()
    train_inputs = task_family.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
    
    for _ in range(4):  # inner-loop update of context via SGD 
        inner_loss = eval_model(model, context, train_inputs, target_fnc)
        context.grad = torch.autograd.grad(inner_loss, context, create_graph= not args.first_order)[0]
        inner_optim.step()
        
    return context 

def get_meta_loss(args, model, task_family, target_fnc):
    meta_loss = 0
    for t in range(args.tasks_per_metaupdate):
        context = inner_update(args, model, task_family['train'], target_fnc)
        test_inputs = task_family['test'].sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)
        meta_loss += eval_model(model, context, test_inputs, target_fnc)
    return meta_loss / args.tasks_per_metaupdate


def meta_backward_1hot(args, model, inner_optimizer, outer_optimizer, task_family, target_functions, labels_1hot):
    # --- compute meta gradient ---    
    train_inputs = task_family['train'].sample_inputs(args.k_meta_train * args.tasks_per_metaupdate, args.use_ordered_pixels).to(args.device)
    test_inputs = task_family['test'].sample_inputs(args.k_meta_train * args.tasks_per_metaupdate, args.use_ordered_pixels).to(args.device)

    # Create targets and labels of all tasks in batch level
    train_targets = create_targets(train_inputs, target_functions, args.tasks_per_metaupdate)
    test_targets = create_targets(test_inputs, target_functions, args.tasks_per_metaupdate)
    labels = create_1hot_labels(labels_1hot, args.tasks_per_metaupdate)
    
    # Encoder / inner update
    update_step_1hot(model, inner_optimizer, train_inputs, train_targets, labels, args.tasks_per_metaupdate)

    # Decoder / outer update
    update_step_1hot(model, outer_optimizer, test_inputs, test_targets, labels, args.tasks_per_metaupdate)


def update_step_1hot(model, optimizer, inputs, targets, labels, n_tasks):
    # Update given optimizer (whether inner or outer)
    optimizer.zero_grad()

    loss = eval_model_1hot(model, inputs, targets, labels)
    loss /= n_tasks
    loss.backward()

    optimizer.step()

    return loss

def create_inputs_targets(inputs, targets, n_tasks, batch_size):
    inputs_batch = inputs.reshape(n_tasks, batch_size, -1)
    targets_batch = targets.reshape(n_tasks, batch_size, -1)

    return torch.cat((inputs_batch, targets_batch), dim=2)


#########################################

def run(args, log_interval=5000, rerun=False):

    path = initial_setting(args, rerun)
    
    task_family = get_task_family(args)
    model = get_model_decoder(args, task_family['train'])
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)
    logger = Logger(model)

    start_time = time.time()
    
    # --- main training loop ---
    for i_iter in range(args.n_iter):
        target_functions = task_family['train'].sample_tasks(args.tasks_per_metaupdate)          # sample tasks
        
        meta_optimiser.zero_grad()
        meta_loss = get_meta_loss(args, model, task_family, target_functions) 
        meta_loss.backward()
        meta_optimiser.step()

        # ------------ logging ------------
        if i_iter % log_interval == 0:
            logger, start_time = update_logger(logger, path, args, model, eval_cavia, task_family, i_iter, start_time)

    return logger

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

def run_vae(args, log_interval=5000, rerun=False):
    path = initial_setting(args, rerun)
    task_family = get_task_family(args)

    # Instantiate algorithm
    from algorithms.vae import VAE
    algorithm = VAE()

    # Build model
    model = algorithm.build_model(args, task_family)

    # Create tensorboard writer
    writer = SummaryWriter('./logs/tb_{}'.format(args.log_name))

    # initialise loggers
    logger = Logger(model)

    # intitialise optimisers
    outer_optimizer = optim.Adam(model.decoder.parameters(), args.lr_meta)
    inner_optimizer = optim.Adam(model.encoder.parameters(), args.lr_inner)

    start_time = time.time()

    # --- main training loop ---

    for i_iter in range(args.n_iter):
        # Sample new mini-batch of tasks from pre-sampled batch of tasks
        target_functions = task_family['train'].sample_tasks_vae(total_num_tasks=args.total_num_tasks, batch_num_tasks=args.tasks_per_metaupdate, batch_size=args.k_meta_train)
        algorithm.meta_backward(args, inner_optimizer, outer_optimizer, task_family, target_functions, writer, i_iter)

        if i_iter % log_interval == 0:
            logger, start_time = update_logger(logger, path, args, copy.deepcopy(model), eval_vae, task_family, i_iter, start_time)        

    return logger


#########################################

def eval_cavia(args, model, task_family, num_updates, n_tasks=100, task_type='train', return_gradnorm=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---
    for t in range(n_tasks):

        # sample a task
        target_fnc = task_family.sample_task()

        context = model.reset_context()
        # ------------ update context on current task ------------
        for _ in range(1, num_updates + 1):
            context = inner_update(args, model, task_family, target_fnc)

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

def eval_1hot(args, model, task_family, num_updates, n_tasks=25, task_type='train', return_gradnorm=False):    
    # logging
    losses = []
    gradnorms = []

    # Optimizer for encoder (context)
    inner_optimizer = optim.SGD(model.encoder.parameters(), args.lr_inner)

    # Sample tasks
    target_functions, labels_1hot = task_family.sample_tasks_1hot(total_num_tasks=n_tasks*4, batch_num_tasks=n_tasks, batch_size=args.k_shot_eval)

    # Only if task is valid or test, reinitialize model and perform inner updates
    if task_type != 'train':
        print('Reinit model {}'.format(task_type))
        # Reinitialize one hot encoder for evaluation mode
        model.encoder.reinit(args.num_context_params, task_family.total_num_tasks)
        model.encoder.train()
        
        # Inner loop steps
        for _ in range(1, num_updates + 1):
            inputs = task_family.sample_inputs(args.k_meta_test * n_tasks, args.use_ordered_pixels).to(args.device)
            targets = create_targets(inputs, target_functions, n_tasks)
            labels = create_1hot_labels(labels_1hot, n_tasks)

            loss = update_step_1hot(model, inner_optimizer, inputs, targets, labels, n_tasks)
            # print('{} loss: {}'.format(task_type, loss))

     # Get entire range's input and respective 1 hot labels
    input_range = task_family.get_input_range().to(args.device).repeat(n_tasks, 1)
    bs = int(input_range.shape[0] / n_tasks)
    input_range_1hot = torch.cat([task_family.create_input_range_1hot_labels(batch_size=bs, cur_label=labels_1hot[t, 0, :]) for t in range(n_tasks)])
    input_range_targets = create_targets(input_range, target_functions, n_tasks)
    
    # compute true loss on entire input range
    model.eval()
    losses.append(eval_model_1hot(model, input_range, input_range_targets, input_range_1hot).item())
    #viz_pred(model, input_range[:bs], input_range_targets[:bs], input_range_1hot[:bs], task_type)
    model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)
    