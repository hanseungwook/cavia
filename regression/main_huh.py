import os
import argparse
import torch
from pytorch_lightning.loggers import TensorBoardLogger # https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/loggers/tensorboard.html

from train_huh import get_Hierarchical_Model, get_Hierarchical_Task  # , run
from utils import set_seed #, Logger
from torch.optim import Adam, SGD


default_save_path = "/nobackup/users/benhuh/Projects/cavia/shared_results"
default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use the GPU if available

###################

def main(hparams):
    set_seed(hparams.seed)  
    logger = TensorBoardLogger(hparams.log_save_path, name=hparams.log_name, version=hparams.v_num) 
    logger.log_hyperparams(hparams) 

    model = get_Hierarchical_Model(hparams, logger)
    print('Pre-sampling dataset')
    supertask = get_Hierarchical_Task(hparams)
    print('Completed pre-sampling')

    print('Starting training')
    # directly get 'test-loss' without pre-training: zero-shot on super-task env.
    loss, outputs = model(hparams.levels, supertask.load('test'), optimizer=Adam, reset=False, return_outputs=False) #True) # grad_clip = hparams.clip ) 
    print('Finished training')

    # run(hparams, model, supertask)         # obsolete

###############

def get_args(*args):
    parser = argparse.ArgumentParser(description='Regression experiments')

    parser.add_argument('--save-path', type=str, default=default_save_path)
    parser.add_argument('--private', dest='save_path', action='store_false')
    
    parser.add_argument('--device',    type=str, default=default_device)     # "cuda:0" or "cpu"
    parser.add_argument('--v_num',     type=int, default=None, help='version number for resuming') #type=str)
    parser.add_argument('--seed',      type=int, default=42)

    parser.add_argument('--task',         type=str, help="Supertasks to solve",)
    parser.add_argument('--architecture', type=int, nargs='+', default=[1, 40, 40, 1], help="Architecture of neural network")
    parser.add_argument('--model-type',   type=str, choices=["CAVIA", "ADDITIVE", "MULTIPLICATIVE", "ADD_MULTIPLICATIVE"], default="CAVIA") 
    
    parser.add_argument('--prefix',       type=str, default="", help="Prefix for logging")
    parser.add_argument('--log-name',     type=str, help="Logging name", default='experiment') #'test::')
    parser.add_argument('--test_intervals',type=int, nargs='+', default=None)
    # parser.add_argument('--log_interval', type=int, nargs='+', default=100)

    parser.add_argument('--lrs',           type=float, nargs='+', default=None) #[0.05, 0.05, 0.001])     # lr for inner-loop, midloop, outerloop
    parser.add_argument('--n_iters',       type=int,   nargs='+', default=None) #[3, 2, 1000])            # max_iter for inner-loop, midloop, outerloop
    parser.add_argument('--for_iters',     type=int,   nargs='+', default=None) #[3, 1, 1])               # number of iterations to use the same data
    parser.add_argument('--n_contexts',    type=int,   nargs='+', default=None) #[2, 1])                  # number of context variables: phi0, phi1 
    parser.add_argument('--k_batch_train', type=int,   nargs='+', default=None) #[100, 25, 2])            # number of datapoints, tasks, super-tasks 
    parser.add_argument('--k_batch_test',  type=int,   nargs='+', default=None) #[100, 25, 2])            # number of datapoints, tasks, super-tasks
    parser.add_argument('--k_batch_valid', type=int,   nargs='+', default=None) #[100, 25, 2])            # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_train', type=int,   nargs='+', default=None) #[30, 15, 2])             # number of datapoints, tasks, super-tasks 
    parser.add_argument('--n_batch_test',  type=int,   nargs='+', default=None) #[30, 15, 2])             # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_valid', type=int,   nargs='+', default=None) #[30, 15, 2])             # number of datapoints, tasks, super-tasks
    
    parser.add_argument('--encoders',      type=str,   nargs='+', default=[None, None, None])      # task encoder-models for model-based Meta-learning. Optimization-based if None (e.g. MAML) 
    
    parser.add_argument('--log_loss_levels', type=int, nargs='+', default=[]) 
    parser.add_argument('--log_ctx_levels',  type=int, nargs='+', default=[]) 
    parser.add_argument('--task_separate_levels', type=int, nargs='+', default=[]) 
    parser.add_argument('--print_loss_levels', type=int, nargs='+', default=[]) 

    parser.add_argument('--higher_flag',   action='store_true', default=False, help='Use Higher optimizer')
    parser.add_argument('--data_parallel', action='store_true', default=False, help='Use data parallel for inner model (decoder)')
    parser.add_argument('--first_order',   action='store_true', default=False, help='run first-order version (create-graph = False)')
    parser.add_argument('--viz',           action='store_true', default=False, help='Run visualize (with pre-trained model)')
    
#     parser.add_argument('--load_model', type=str,     default='',     help='Path to model weights to load')


    parser_ = parser  #  = MODEL.add_model_specific_args(parser)
    args, unknown = parser_.parse_known_args(*args) 
    
    args = check_hparam_default(args)
    return args


###################

def check_hparam_default(hparams):
    ## Default copy replacing None
    
    hparams.save_path = hparams.save_path or os.getcwd()
    hparams.log_save_path = os.path.join(hparams.save_path, "logs")
    hparams.log_name = hparams.log_name or hparams.task  #+'_test'
    
    if not os.path.exists(hparams.log_save_path):
        os.makedirs(hparams.log_save_path)
    
    hparams.levels         = len(hparams.n_contexts) + 1 #len(decoder_model.parameters_all) #- 1
    
    hparams.for_iters     = hparams.for_iters or [1]*hparams.levels
    hparams.lrs           = hparams.lrs       or [0.01]*hparams.levels
    hparams.test_intervals= hparams.test_intervals or [100]*hparams.levels
    
    # hparams.log_loss_levels = hparams.log_loss_levels or [] # [False]*hparams.levels #*(hparams.levels+1)
    # hparams.log_ctx_levels  = hparams.log_ctx_levels or [] #[False]*hparams.levels
    # hparams.task_separate_levels = hparams.task_separate_levels or [] # [False]*hparams.levels

    hparams.k_batch_train = hparams.k_batch_train or [None]*hparams.levels # ## maybe duplicate k_batch_train/n_batch_train 
    hparams.k_batch_test  = hparams.k_batch_test  or [None]*hparams.levels # hparams.k_batch_train
    hparams.k_batch_valid = hparams.k_batch_valid or [None]*hparams.levels # hparams.k_batch_train

    hparams.n_batch_train = hparams.n_batch_train or hparams.k_batch_train
    hparams.n_batch_test  = hparams.n_batch_test  or hparams.k_batch_test  
    hparams.n_batch_valid = hparams.n_batch_valid or hparams.k_batch_valid 
    
    
    for name in ['lrs', 'n_iters', 'for_iters', 'k_batch_train', 'k_batch_test', 'k_batch_valid', 'n_batch_train', 'n_batch_test', 'n_batch_valid']:
        temp = getattr(hparams,name)
        assert temp is None or len(getattr(hparams,name)) == hparams.levels, "hparams."+name+" has wrong length"
        
    return hparams



###################

if __name__ == '__main__':
    hparams = get_args()
    main(hparams)