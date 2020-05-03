import logging
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


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
    colors = [
        sns.color_palette("hls", 8)[0],
        sns.color_palette("hls", 8)[2],
        sns.color_palette("hls", 8)[5],
        sns.color_palette("hls", 8)[7]] 

    # TODO Consider same PCA dimension
    pca = PCA(n_components=2)

    contexts = torch.stack(higher_contexts).detach().cpu().numpy()
    for lower_contexts_ in lower_contexts:
        lower_contexts_ = torch.stack(lower_contexts_).detach().cpu().numpy()
        contexts = np.concatenate((contexts, lower_contexts_))
    contexts = pca.fit_transform(contexts)
    higher_contexts = contexts[0:4, :]
    lower_contexts = contexts[4:, :]

    # Visualize higher contexts
    for i_super_task in range(4):
        x, y = higher_contexts[i_super_task, :] 
        plt.scatter(x, y, c=colors[i_super_task], s=65)

    # Visualize lower contexts
    for i_super_task in range(4):
        for i_task in range(25):
            i_context = i_super_task * 25 + i_task
            x, y = lower_contexts[i_context, :] 
            plt.scatter(x, y, c=colors[i_super_task], alpha=0.25, s=30)

    plt.legend()
    plt.title("PCA_iteration" + str(iteration))
    # plt.xlim([-1., 1.])
    # plt.ylim([-1., 1])
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
