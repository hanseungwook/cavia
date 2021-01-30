import logging
import random
import torch
import numpy as np
import gym
from tensorboardX import SummaryWriter
from model import get_model_type
from misc.rl_utils import make_env


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.log = set_log(args)
        self.tb_writer = SummaryWriter('./log/tb_{0}'.format(args.log_name))

    def update(self, key, value, iter):
        self.log[self.args.log_name].info("At iteration {}, {}: {:.3f}".format(iter, key, value))
        self.tb_writer.add_scalar(key, value, iter)


def set_log(args):
    log = {}
    set_logger(logger_name=args.log_name, log_file=r'{0}{1}'.format("./log/", args.log_name))  
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
    model_type = get_model_type(args.model_type, is_rl=True)
    logger.log[args.log_name].info("Selecting base_model: {}".format(model_type))

    # Overwrite last layer of the architecture according to the action space of the environment
    # Note that we put a default env and task only to get the action space of the environment
    env = gym.make("AntVel-v1")
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.architecture[0] = input_dim
    args.architecture[-1] = action_dim
    env.close()

    model = model_type(n_arch=args.architecture, n_contexts=args.n_contexts, is_continuous_action=args.is_continuous_action, 
                        device=args.device).to(args.device)
    logger.log[args.log_name].info("Model: {}".format(model))

    return model
