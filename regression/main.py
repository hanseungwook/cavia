import os
import arguments
from hierarchical import Hierarchical_Model, get_hierarchical_task
from misc.utils import set_seed, Logger, make_batch_dict, get_base_model
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

    # Get hierarchical task
    batch_dict = make_batch_dict(args.batch)
    task = get_hierarchical_task(batch_dict, args)

    # set hierarchical model
    base_model = get_base_model(args, logger)
    model = Hierarchical_Model(
        base_model=base_model, 
        args=args, 
        task=task,
        logger=logger)

    # Start train
    model(task, optimizer=Adam, reset=False, is_outer=False)
