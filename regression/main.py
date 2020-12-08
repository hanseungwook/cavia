import os
import arguments
from hierarchical import Hierarchical_Model, get_hierarchical_task
from misc.utils import set_seed, Logger, get_base_model
from torch.optim import Adam


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create logger
    if not os.path.exists("./log"):
        os.makedirs("./log")
    logger = Logger(args=args)

    # set hierarchical model
    base_model = get_base_model(args, logger)
    model = Hierarchical_Model(
        base_model=base_model, 
        args=args, 
        logger=logger)

    # Get hierarchical task
    task = get_hierarchical_task(args, logger)

    # Start train
    while True:
        # Train one outer-loop
        model(task, optimizer=Adam, reset=False)

        # Get new task
        task[0].sample_new_tasks(task, args.batch)
