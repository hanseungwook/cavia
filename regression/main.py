import os
import arguments
from train import run
from utils import set_seed, Logger


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    logger = Logger(args=args)

    # Start train
    run(args, logger)
