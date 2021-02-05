import argparse

# from arguments import run_argparse
from train_huh import run #, get_loggers
import os
from utils import set_seed, Logger
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger # https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/loggers/tensorboard.html

from pdb import set_trace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use the GPU if available

def main(hparams):
    # Load arguments

    save_path = "/nobackup/users/benhuh/Projects/cavia/shared_results"
    log_save_path = os.path.join(save_path, "logs")
    hparams.log_save_path = log_save_path
    if hparams.log_name is None:
         hparams.log_name = hparams.task[0]+'_test'
    log_name = hparams.log_name 
    
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    set_seed(hparams.seed)  
    logger = TensorBoardLogger(log_save_path, name=log_name, version=hparams.v_num) 
    logger.log_hyperparams(hparams)
#     set_trace()
#     logger2 = SummaryWriter(log_save_path+'/'+log_name)
#     logger2.add_hparams(var(hparams), {"metric/distance": 0.5})
    
    run(hparams, logger)         # Start train

    
###############





def get_args(jupyter_flag = False):
    parser = argparse.ArgumentParser(description='Regression experiments')

    parser.add_argument('--task', type=str,  nargs='+', help="Supertasks to solve",)
#                         choices=['sine', 'linear', 'celeba', 'cifar10', 'hier-imagenet', 'celeba_airplane', 'airplane', 'mnist', 'fashion_mnist', 
#                                  'mnist_fmnist', 'mnist_fmnist_3level'])
    parser.add_argument('--classes', type=int, nargs='+', default=[], help="Specified classes of dataset to use (if not specified, all)")
    parser.add_argument('--architecture', type=int, nargs='+', default=[1, 40, 40, 1], help="Architecture of neural network")
    parser.add_argument('--first_order', action='store_true', default=False, help='run first-order version (create-graph = False)')
    parser.add_argument('--model-type', type=str, choices=["CAVIA", "ADDITIVE", "MULTIPLICATIVE", "ADD_MULTIPLICATIVE"], default="CAVIA") 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix', type=str, default="", help="Prefix for logging")

    parser.add_argument('--log-name', nargs='?', type=str, help="Logging name", default=None) #'test::')
    parser.add_argument('--log_interval',   type=int, default=100)
    parser.add_argument('--test_interval',  type=int, default=0)
    parser.add_argument('--v_num', type=int, help='version number for resuming') #type=str)

    parser.add_argument('--viz', action='store_true', default=False,  help='Run visualize (with pre-trained model)')
    parser.add_argument('--load_model', type=str, default='',         help='Path to model weights to load')
    parser.add_argument('--data_parallel', action='store_true', default=False, help='Use data parallel for inner model (decoder)')

    parser.add_argument('--lrs',           type=float, nargs='+', default=[0.05, 0.05, 0.001])   # lr  for inner-loop, midloop, outerloop
    parser.add_argument('--n_iters',       type=int, nargs='+', default=[3, 2, 1000])            # max_iter for inner-loop, midloop, outerloop
    parser.add_argument('--for_iters',     type=int, nargs='+', default=[3, 1, 1])               # number of iterations to use the same data
    parser.add_argument('--k_batch_train', type=int, nargs='+', default=[100, 25, 2])            # number of datapoints, tasks, super-tasks 
    parser.add_argument('--k_batch_test',  type=int, nargs='+', default=[100, 25, 2])            # number of datapoints, tasks, super-tasks
    parser.add_argument('--k_batch_valid', type=int, nargs='+', default=[100, 25, 2])            # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_train', type=int, nargs='+', default=[30, 15, 2])             # number of datapoints, tasks, super-tasks 
    parser.add_argument('--n_batch_test',  type=int, nargs='+', default=[30, 15, 2])             # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_valid', type=int, nargs='+', default=[30, 15, 2])             # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_contexts',    type=int, nargs='+', default=[2, 1])                  # number of context variables: phi0, phi1 
    parser.add_argument('--encoders',      type=int, nargs='+', default=[None, None, None])      # task encoder-models for model-based Meta-learning. Optimization-based if None (MAML) 
    
    parser.add_argument('--ctx_logging_levels', type=int, nargs='+', default=[]) 
    parser.add_argument('--higher_flag',  action='store_true', default=False, help='Use Higher optimizer')
    
    parser.add_argument('--device',      type=str, default=device)     # "cuda:0" or "cpu"

    parser_ = parser  #  = MODEL.add_model_specific_args(parser)
    args, unknown = parser_.parse_known_args("") if jupyter_flag else parser_.parse_known_args()   # needs "" input to run from jupyter notebook.

    

    return args


if __name__ == '__main__':
    hparams = get_args()
    main(hparams)