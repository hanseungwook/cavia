import logging
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#################################################################################
# LOGGING
#################################################################################
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


def set_log(args):
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    return log


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


#################################################################################
# VISUALIZATION
#################################################################################
def vis_pca(higher_contexts, lower_contexts, task_family, iteration, args):
    # Create directories
    if not os.path.exists("./logs/n_inner" + str(args.n_inner)):
        os.makedirs("./logs/n_inner" + str(args.n_inner))

    # Set color for visualization
    n_super_task = len(task_family.super_tasks)
    colors = []
    for i_super_task in range(n_super_task):
        colors.append(sns.color_palette("hls", n_super_task)[i_super_task])

    # Preprocess data
    lower_contexts_ = []
    for lower_context in lower_contexts:
        lower_context = torch.stack(lower_context).detach().cpu().numpy()
        lower_contexts_.append(lower_context)

    contexts = np.vstack(lower_contexts_)
    if args.n_context_models == 2:
        raise NotImplementedError()
        higher_contexts = torch.stack(higher_contexts).detach().cpu().numpy()
        contexts = np.concatenate((contexts, higher_contexts))
    # contexts = StandardScaler().fit_transform(contexts)

    # Fit PCA
    pca = PCA(n_components=2)
    contexts = pca.fit_transform(contexts)
    for i_super_task, context in enumerate(np.split(contexts, n_super_task)):
        for i_task in range(args.tasks_per_metaupdate):
            x, y = context[i_task, :] 
            plt.scatter(x, y, c=colors[i_super_task], alpha=1.0, s=30)

    plt.legend()
    plt.title("PCA_iteration" + str(iteration))
    # plt.xlim([-1., 1.])
    # plt.ylim([-1., 1])
    plt.savefig("logs/n_inner" + str(args.n_inner) + "/pca_iteration" + str(iteration).zfill(3) + ".png")
    plt.close()


def vis_prediction(model, higher_contexts, lower_contexts, val_data, iteration, args):
    # Create directories
    if not os.path.exists("./logs/n_inner" + str(args.n_inner)):
        os.makedirs("./logs/n_inner" + str(args.n_inner))

    for i_super_task, super_task in enumerate(val_data):
        higher_context = higher_contexts[i_super_task]

        for i_task, task in enumerate(super_task):
            lower_context = lower_contexts[i_super_task][i_task]
            input, target = task
            pred = model(input, lower_context, higher_context).detach().numpy()
            break

        plt.figure()
        plt.scatter(input, pred, label="pred")
        plt.scatter(input, target, label="gt")
        plt.legend()
        plt.title(str(i_super_task) + "_iteration" + str(iteration))

        plt.savefig(
            "logs/n_inner" + str(args.n_inner) + "/iteration" + 
            str(iteration).zfill(3) + "_" + str(i_super_task) + ".png")
        plt.close()


def vis_context(lower_contexts, task_family, iteration, args):
    # Create directories
    if not os.path.exists("./logs/n_inner" + str(args.n_inner)):
        os.makedirs("./logs/n_inner" + str(args.n_inner))

    # Set color for visualization
    n_super_task = len(task_family.super_tasks)
    assert n_super_task == 1, "Should be only sinusoidal task"

    # Preprocess data
    context = torch.stack(lower_contexts[0]).detach().cpu().numpy()

    # Visualize
    x, y = context[:, 0], context[:, 1]
    plt.scatter(x, y, c=task_family.phases, s=30)
    plt.colorbar()
    plt.title("Context (phase) iteration" + str(iteration))
    plt.savefig("logs/n_inner" + str(args.n_inner) + "/context_iteration" + str(iteration).zfill(3) + "_phase.png")
    plt.close()

    plt.scatter(x, y, c=task_family.amplitudes, s=30)
    plt.colorbar()
    plt.title("Context (amp) iteration" + str(iteration))
    plt.savefig("logs/n_inner" + str(args.n_inner) + "/context_iteration" + str(iteration).zfill(3) + "_amp.png")
    plt.close()
