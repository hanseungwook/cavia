import os
import argparse
import torch
from pytorch_lightning.loggers import TensorBoardLogger # https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/loggers/tensorboard.html

from hierarchical_eval import update_status
from train_huh import get_batch_dict, save_model, load_model, get_save_dir, get_base_model, get_encoder_model, Hierarchical_Eval  #get_Hierarchical_Eval
from hierarchical_task import Hierarchical_Task, Task_sampler
from make_tasks import get_task

from utils import set_seed #, Logger
from torch.optim import Adam, SGD

from pdb import set_trace

default_save_path = "/nobackup/users/benhuh/Projects/cavia/shared_results"
default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use the GPU if available

###################

def main(hparams, model_loss = None, task = None):
    set_seed(hparams.seed)  
    save_dir = get_save_dir(hparams)    
    base_model, base_loss  = get_base_model(hparams) 
    base_loss = model_loss or base_loss
    
    if hparams.v_num is not None: 
        base_model = load_model(base_model, save_dir, hparams.log_name, hparams.v_num)  # fix load_model ! 
        
    encoder_models  = None # get_encoder_model(hparams.encoders, hparams)                       # adaptation model: None == MAML
    
    logger = TensorBoardLogger(hparams.log_save_path, name=hparams.log_name, version=hparams.v_num) 
    logger.log_hyperparams(hparams) 
    
    evaluator = Hierarchical_Eval(hparams, base_model, encoder_models, base_loss, logger)

#     print('generating Hierarchical Task') 
    task  = task or get_task(hparams.task, hparams.task_args)
    task_hierarchy = Hierarchical_Task(task, batch_dict=get_batch_dict(hparams))  # get_Hierarchical_Task(hparams)

    print('start meta-evaluation')     # evaluate 'test-loss' on super-task without training.
    print('tasks:', task[1])    
    loss, outputs = evaluator.evaluate(task_hierarchy, task_idx = hparams.task, optimizer=Adam, reset=False, return_outputs=False) #  loss, outputs = evaluator(task.load('test'), sample_type='test', optimizer=Adam, reset=False, return_outputs=False)

    logger.save()
    save_model()  # fix save_model()!  also load_model()
    print('Finished training and saving logger')

###############

def get_args(*args):
    parser = argparse.ArgumentParser(description='Regression experiments')

    parser.add_argument('--save-path', type=str, default=default_save_path)
    parser.add_argument('--private', dest='save_path', action='store_false')
    parser.add_argument('--prefix',       type=str, default="", help="Prefix for logging")
    parser.add_argument('--log-name',     type=str, help="Logging name", default=None) #'experiment') #'test::')
    
    parser.add_argument('--task',         type=str, help="Supertasks to solve",)
    parser.add_argument('--task_args',    type=int, nargs='+', default=[],  help="input to task_fnc")
    parser.add_argument('--model-type',   type=str, choices=["CAVIA", "ADDITIVE", "MULTIPLICATIVE", "ADD_MULTIPLICATIVE"], default="CAVIA") 
    parser.add_argument('--architecture', type=int, nargs='+', default=[1, 40, 40, 1], help="Architecture of neural network")
    parser.add_argument('--n_contexts',   type=int, nargs='+', default=[], help="number of context variables: phi0, phi1 ") #[2, 1])   
    
    parser.add_argument('--device',    type=str, default=default_device)     # "cuda:0" or "cpu"
    parser.add_argument('--v_num',     type=int, default=None, help='version number for resuming') #type=str)
    parser.add_argument('--seed',      type=int, default=42)

    parser.add_argument('--lrs',       type=float, nargs='+', default=None, help="lr for lv0/lv1/lv2..") #[0.05, 0.05, 0.001])     # lr for inner-loop, midloop, outerloop
    parser.add_argument('--max_iters', type=int,   nargs='+', default=None, help="max_iter for lv0/lv1/lv2..") #[3, 2, 1000])            # max_iter for inner-loop, midloop, outerloop
    parser.add_argument('--for_iters', type=int,   nargs='+', default=None) #[3, 1, 1])               # number of iterations to use the same data
    
    parser.add_argument('--k_train',   type=int,   nargs='+', default=None, help="total-batch for lv0/lv1/lv2. 'k-shot' learning.") #[100, 25, 2])  
    parser.add_argument('--k_test',    type=int,   nargs='+', default=None, help="total-batch for lv0/lv1/lv2.")
    parser.add_argument('--k_valid',   type=int,   nargs='+', default=None, help="total-batch for lv0/lv1/lv2.")
    parser.add_argument('--n_train',   type=int,   nargs='+', default=None, help="minibatch for lv0/lv1/lv2.") #[30, 15, 2])     
    parser.add_argument('--n_test',    type=int,   nargs='+', default=None, help="minibatch for lv0/lv1/lv2.")
    parser.add_argument('--n_valid',   type=int,   nargs='+', default=None, help="minibatch for lv0/lv1/lv2.")

    parser.add_argument('--grad_clip', type=int,     default=100, help="grad_clip_value") #[30, 15, 2])          

    parser.add_argument('--encoders',  type=str,   nargs='+', default=[None, None, None])      # task encoder-models for model-based Meta-learning. Optimization-based if None (e.g. MAML) 
    
    parser.add_argument('--log_intervals', type=int, nargs='+', default=None)
    parser.add_argument('--test_intervals',type=int, nargs='+', default=None)

    parser.add_argument('--log_loss_levels',      type=int, nargs='+', default=None) 
    parser.add_argument('--log_ctx_levels',       type=int, nargs='+', default=None) 
    parser.add_argument('--task_separate_levels', type=int, nargs='+', default=None) 
    parser.add_argument('--print_levels',    type=int, nargs='+', default=None) 

    parser.add_argument('--use_higher',    action='store_true', default=False, help='Use Higher optimizer')
    parser.add_argument('--data_parallel', action='store_true', default=False, help='Use data parallel for inner model (decoder)')
    parser.add_argument('--mp', action='store_true', default=False, help='Use multiprocessing for CPU (only)')
    parser.add_argument('--first_order',   action='store_true', default=False, help='run first-order version (create-graph = False)')
    parser.add_argument('--viz',           action='store_true', default=False, help='Run visualize (with pre-trained model)')

    parser.add_argument('--profile',       action='store_true', default=False, help='Run profiling')

    
#     parser.add_argument('--load_model', type=str,     default='',     help='Path to model weights to load')
    args, unknown = parser.parse_known_args(*args) 
    
    args = check_hparam_default(args)
    return args


###################

def check_hparam_default(hparams):
    ## Default copy replacing None

    if hparams.task == 'LQR':
        hparams.model_type = None
        hparams.architecture = None
    
    hparams.save_path = hparams.save_path or os.getcwd()
    hparams.log_save_path = os.path.join(hparams.save_path, "logs")
    hparams.log_name = hparams.log_name or hparams.task  #+'_test'
    
    if not os.path.exists(hparams.log_save_path):
        os.makedirs(hparams.log_save_path)
    
    hparams.top_level     = len(hparams.n_contexts) #+ 1 #len(decoder_model.parameters_all) #- 1
    
    hparams.for_iters     = hparams.for_iters or [1]*(hparams.top_level+1)
    hparams.lrs           = hparams.lrs       or [0.01]*(hparams.top_level+1)
    hparams.log_intervals = hparams.log_intervals or [1]*(hparams.top_level+1)
    hparams.test_intervals= hparams.test_intervals or [100]*(hparams.top_level+1)

    # hparams.grad_clip = grad_clip or [100]*hparams.top_level
    
    hparams.log_loss_levels      = hparams.log_loss_levels or [hparams.top_level] # [False]*hparams.top_level #*((hparams.top_level+1))
    hparams.log_ctx_levels       = hparams.log_ctx_levels or [hparams.top_level] #[False]*hparams.top_level
    hparams.task_separate_levels = hparams.task_separate_levels or [hparams.top_level, hparams.top_level-1] # [False]*hparams.top_level
    hparams.print_levels         = hparams.print_levels or [hparams.top_level]

    hparams.k_train = hparams.k_train or [None]*(hparams.top_level+1) # ## maybe duplicate k_train/n_train 
    hparams.k_test  = hparams.k_test  or [None]*(hparams.top_level+1) # hparams.k_train
    hparams.k_valid = hparams.k_valid or [None]*(hparams.top_level+1) # hparams.k_train

    hparams.n_train = hparams.n_train or hparams.k_train
    hparams.n_test  = hparams.n_test  or hparams.k_test  
    hparams.n_valid = hparams.n_valid or hparams.k_valid 
    
    
    for name in ['lrs', 'max_iters', 'for_iters', 'k_train', 'k_test', 'k_valid', 'n_train', 'n_test', 'n_valid', 'log_intervals', 'test_intervals']:
        temp = getattr(hparams,name)
        assert temp is None or len(getattr(hparams,name)) == (hparams.top_level+1), "hparams."+name+" has wrong length"
        
#     hparams.log_intervals += [1]  # for top+1 super-level
    return hparams



###################

if __name__ == '__main__':
    hparams = get_args()

    if hparams.profile:
        print('profiling')

        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        main(args)
        profiler.disable()
        profiler.sort_stats('cumtime')
        profiler.dump_stats('profile_output.prof')
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

    else:
        # print('not profiling')
        main(hparams)
