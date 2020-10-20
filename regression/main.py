import os
import arguments
from utils import set_seed, Logger, make_batch_dict, get_base_model
from hierarchical import Hierarchical_Model, get_hierarchical_task


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    logger = Logger(args=args)

    # Get hierarchical task
    batch_dict = make_batch_dict(args.k_batch_train, args.k_batch_test, args.k_batch_valid)
    task = get_hierarchical_task(
        task_func_list=["MiniGrid-Empty-5x5-v0", "MiniGrid-Unlock-v0"], 
        batch_dict=batch_dict)

    # Start train
    base_model = get_base_model(args, logger)
    model = Hierarchical_Model(
        decoder_model=base_model, 
        encoders=None, 
        args=args, 
        task=task)

    raise ValueError("Define TRPO optimizer here")

    import sys
    sys.exit()
    model(task, reset=False)
