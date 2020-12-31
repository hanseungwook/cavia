import os
import arguments
from trainer import train
from hierarchical.model import get_base_model
from hierarchical.task import HierarchicalTask
from misc.utils import set_seed, Logger


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
    hierarchical_task = HierarchicalTask(args, logger)

    # Start train
    train(base_model, hierarchical_task, args, logger)
