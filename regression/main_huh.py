import arguments
from train_huh import run
import os
from utils import set_seed, Logger
from functools import partial


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    set_seed(args.seed)       # Set random seed
    # logger  = Logger(args)    # Set log
    logger  = partial(Logger, args)    # Set log
    run(args, logger)         # Start train
