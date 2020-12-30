import logging
import random
import torch
import gym
import numpy as np
from tensorboardX import SummaryWriter
from model.cavia import CAVIA
from misc.rl_utils import make_env


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.log = set_log(args)
        self.tb_writer = SummaryWriter("./log/tb_{0}".format(args.log_name))

    def update(self, key, value, iter):
        self.log[self.args.log_name].info("At iteration {}, {}: {:.3f}".format(iter, key, value))
        self.tb_writer.add_scalar(key, value, iter)


def set_log(args):
    log = {}
    set_logger(logger_name=args.log_name, log_file=r"{0}{1}".format("./log/", args.log_name))
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
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def get_base_model(args, logger):
    # Overwrite last layer of the architecture according to the action space of the environment
    # Note that we put a default env and task only to get the action space of the environment
    env = make_env(args=args)()
    input_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.box.Box):
        args.is_continuous_action = True
        action_dim = env.action_space.shape[0]
    else:
        args.is_continuous_action = False
        action_dim = env.action_space.n
    args.network_arch[0] = input_dim
    args.network_arch[-1] = action_dim
    env.close()

    # Overwrite input layer of the architecture with number of context parameters
    if args.is_hierarchical_learning:
        args.network_arch[0] += sum(args.n_contexts)
    else:
        args.network_arch[0] += args.n_contexts[0]

    # Return base_model
    base_model = CAVIA(args, logger)
    logger.log[args.log_name].info("Model: {}".format(base_model))
    return base_model
