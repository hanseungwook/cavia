import copy
import os
import time
import torch
import utils
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torch.optim as optim
from logger import Logger
from tensorboardX import SummaryWriter


def initial_setting(args, rerun):
    # Check if we already ran this experiment
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
        from regression.task import sine
        task_family['train'] = sine.RegressionTasksSinusoidal()
        task_family['valid'] = sine.RegressionTasksSinusoidal()
        task_family['test'] = sine.RegressionTasksSinusoidal()
    elif args.task == 'mixture':
        from regression.task import mixture
        task_family['train'] = mixture.MixutureRegressionTasks(args)
        task_family['valid'] = mixture.MixutureRegressionTasks(args)
        task_family['test'] = mixture.MixutureRegressionTasks(args)
    else:
        raise NotImplementedError()

    return task_family


def get_model_decoder(args):
    n_arch = args.architecture
    n_context = args.n_context_params

    if args.model_type == "CAVIA":
        from cavia_model import CaviaModel
        MODEL = CaviaModel
        n_arch[0] += n_context * args.n_context_models
    else:
        raise ValueError()
        
    model_decoder = MODEL(
        n_arch=n_arch, 
        n_context=n_context, 
        device=args.device).to(args.device)
    return model_decoder


def update_logger(logger, path, args, model, eval_fnc, task_family, i_iter, start_time):

    def logger_helper(logger, args, model, eval_fnc, task_family, task_type):
        loss_mean, loss_conf = eval_fnc(
            args, copy.deepcopy(model), task_family=task_family[task_type], 
            num_updates=args.num_inner_updates)
        getattr(logger, task_type + '_loss').append(loss_mean)
        getattr(logger, task_type + '_conf').append(loss_conf)
        return logger

    # Evaluate on training set
    logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='train')
    logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='test')
    logger = logger_helper(logger, args, model, eval_fnc, task_family, task_type='valid')

    # Save logging results
    utils.save_obj(logger, path)

    # Save best model
    if logger.valid_loss[-1] == np.min(logger.valid_loss):
        logger.update_best_model(model)
        print('saving best model at iter', i_iter)

    # Print current results
    logger.print_info(i_iter, start_time)
    start_time = time.time()

    return logger, start_time


def eval_model(model, lower_context, higher_context, inputs, target_fnc):
    outputs = model(inputs, lower_context, higher_context)
    targets = target_fnc(inputs)
    return F.mse_loss(outputs, targets)


def inner_update_for_lower_context(args, model, task_family, task_function, higher_context):
    # TODO Support multiple inner-loop
    lower_context = model.reset_context()
    # lower_optimizer = optim.SGD([lower_context], args.lr_inner)
    train_inputs = task_family['train'].sample_inputs(args.k_meta_train).to(args.device)
    
    lower_inner_loss = eval_model(model, lower_context, higher_context, train_inputs, task_function)
    # lower_context.grad = torch.autograd.grad(lower_inner_loss, lower_context, create_graph=True)[0]
    lower_context_grad = torch.autograd.grad(lower_inner_loss, lower_context, create_graph=True)[0]
    lower_context = lower_context - lower_context_grad * args.lr_inner
    # lower_optimizer.step()
        
    return lower_context 


def get_meta_loss(args, model, task_family):
    # TODO Support multiple inner-loop
    meta_losses = []

    for super_task in task_family["train"].super_tasks:
        task_functions = task_family["train"].sample_tasks(super_task)
        higher_context = model.reset_context()
        # higher_optimizer = optim.SGD([higher_context], args.lr_inner)

        higher_inner_losses, lower_contexts = [], []

        for i_task in range(args.tasks_per_metaupdate):
            task_function = task_functions[i_task]
            lower_context = inner_update_for_lower_context(
                args, model, task_family, task_function, higher_context)
            lower_contexts.append(lower_context)

            # Compute inner-loop loss for higher context based on adapted lower context
            train_inputs = task_family['train'].sample_inputs(args.k_meta_train).to(args.device)
            higher_inner_losses.append(eval_model(model, lower_context, higher_context, train_inputs, task_function))

        # Inner-loop update for higher context
        higher_inner_loss = sum(higher_inner_losses) / float(len(higher_inner_losses))
        # higher_context.grad = torch.autograd.grad(higher_inner_loss, higher_context, create_graph=True)[0]
        higher_context_grad = torch.autograd.grad(higher_inner_loss, higher_context, create_graph=True)[0]
        higher_context = higher_context - higher_context_grad * args.lr_inner
        # higher_optimizer.step()

        # Get meta-loss for the base model
        for i_task in range(args.tasks_per_metaupdate):
            task_function = task_functions[i_task]
            lower_context = lower_contexts[i_task] 
            test_inputs = task_family['test'].sample_inputs(args.k_meta_train).to(args.device)
            meta_losses.append(eval_model(model, lower_context, higher_context, test_inputs, task_function))

    return sum(meta_losses) / float(len(meta_losses))


def run(args, log_interval=5000, rerun=False):
    path = initial_setting(args, rerun)
    tb_writer = SummaryWriter('./logs/tb_{0}'.format("tmp"))
    
    # Set tasks
    task_family = get_task_family(args)

    # Set model
    model = get_model_decoder(args)
    meta_optimizer = optim.Adam(model.parameters(), args.lr_meta)

    # Begin meta-train
    logger = Logger(model)
    start_time = time.time()
    
    for i_iter in range(args.n_iter):
        meta_optimizer.zero_grad()
        meta_loss = get_meta_loss(args, model, task_family) 
        meta_loss.backward()
        meta_optimizer.step()

        tb_writer.add_scalar("Meta loss:", meta_loss.detach().cpu().numpy(), i_iter)

        # # ------------ logging ------------
        # if i_iter % log_interval == 0:
        #     logger, start_time = update_logger(logger, path, args, model, eval_cavia, task_family, i_iter, start_time)

    return logger


def eval_cavia(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False):
    # get the task family

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

            # Disabling tracking grad norms for now
            # gradnorms.append(task_gradients[0].norm().item())

        # compute true loss on entire input range
        model.eval()
        losses.append(eval_model(model, context, curr_inputs, target_fnc).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)
