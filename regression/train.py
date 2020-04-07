import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import set_log
from logger import Logger
from tensorboardX import SummaryWriter


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


def eval_model(model, lower_context, higher_context, inputs, target_fnc):
    outputs = model(inputs, lower_context, higher_context)
    targets = target_fnc(inputs)
    return F.mse_loss(outputs, targets)


def inner_update_for_lower_context(args, model, task_family, task_function, higher_context):
    # TODO Support multiple inner-loop
    lower_context = model.reset_context()
    train_inputs = task_family['train'].sample_inputs(args.k_meta_train).to(args.device)
    
    lower_inner_loss = eval_model(model, lower_context, higher_context, train_inputs, task_function)
    lower_context_grad = torch.autograd.grad(lower_inner_loss, lower_context, create_graph=True)[0]
    lower_context = lower_context - lower_context_grad * args.lr_inner
        
    return lower_context 


def get_meta_loss(model, task_family, args, log, tb_writer, iteration):
    # TODO Support multiple inner-loop
    meta_losses = []

    for super_task in task_family["train"].super_tasks:
        task_functions = task_family["train"].sample_tasks(super_task)
        higher_context = model.reset_context()

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
        higher_context_grad = torch.autograd.grad(higher_inner_loss, higher_context, create_graph=True)[0]
        higher_context = higher_context - higher_context_grad * args.lr_inner

        log[args.log_name].info("At iteration {} with super task {}, higher-loss: {:.3f}".format(
            iteration, super_task, higher_inner_loss.detach().cpu().numpy()))
        tb_writer.add_scalars(
            "higher_inner_loss", {super_task: higher_inner_loss.detach().cpu().numpy()}, iteration)

        # Get meta-loss for the base model
        for i_task in range(args.tasks_per_metaupdate):
            task_function = task_functions[i_task]
            lower_context = lower_contexts[i_task] 
            test_inputs = task_family['test'].sample_inputs(args.k_meta_train).to(args.device)
            meta_losses.append(eval_model(model, lower_context, higher_context, test_inputs, task_function))

        # Visualize prediction
        if iteration % 100 == 0: 
            vis_prediction(model, lower_context, higher_context, test_inputs, task_function)

    return sum(meta_losses) / float(len(meta_losses))


def run(args, log_interval=5000, rerun=False):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Set log
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    
    # Set tasks
    task_family = get_task_family(args)

    # Set model
    model = get_model_decoder(args)
    meta_optimizer = optim.Adam(model.parameters(), args.lr_meta)

    # Begin meta-train
    logger = Logger(model)
    
    for iteration in range(args.n_iter):
        meta_optimizer.zero_grad()
        meta_loss = get_meta_loss(model, task_family, args, log, tb_writer, iteration) 
        meta_loss.backward()
        meta_optimizer.step()

        log[args.log_name].info("At iteration {}, meta-loss: {:.3f}".format(
            iteration, meta_loss.detach().cpu().numpy()))
        tb_writer.add_scalar("Meta loss:", meta_loss.detach().cpu().numpy(), iteration)

    return logger


def vis_prediction(model, lower_context, higher_context, inputs, task_function):
    outputs = model(inputs, lower_context, higher_context).detach().cpu().numpy()
    targets = task_function(inputs).detach().cpu().numpy()

    plt.figure()
    plt.scatter(inputs, outputs, label="pred")
    plt.scatter(inputs, targets, label="gt")
    plt.legend()
    plt.show()
