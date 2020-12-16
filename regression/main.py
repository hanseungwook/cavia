import os
import arguments
from trainer import train
from task import get_hierarchical_task
from misc.utils import set_seed, Logger, get_base_model


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create logger
    if not os.path.exists("./log"):
        os.makedirs("./log")
    logger = Logger(args=args)

    # Set hierarchical model
    base_model = get_base_model(args, logger)

    # Set hierarchical task
    hierarchical_task = get_hierarchical_task(args, logger)

    # Start train
    train(base_model, hierarchical_task, args, logger)
