import logging
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter


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


def log_result(reward, iteration, args, logger, prefix=""):
    if isinstance(reward, list):
        reward = sum(reward) / float(len(reward))
    logger.log[args.log_name].info("{} reward: {:.3f} at iteration {}".format(prefix, reward, iteration))
    logger.tb_writer.add_scalars("reward", {prefix: reward}, iteration)
