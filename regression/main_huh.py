from arguments import run_argparse
from train_huh import run, get_loggers
import os
from utils import set_seed, Logger
from functools import partial


if __name__ == '__main__':
    # Load arguments
    args = run_argparse()

    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    set_seed(args.seed)  
    logger_maker  = partial(Logger, args) 
    loggers, test_loggers = get_loggers(logger_maker, levels = len(args.k_batch_train))
    
    run(args, loggers, test_loggers)         # Start train
