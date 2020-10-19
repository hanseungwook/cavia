import os
import arguments
from train import run
from utils import set_seed, Logger
from functools import partial


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    logger = partial(Logger, args)    # Set log

    # Start train
    run(args, logger)
