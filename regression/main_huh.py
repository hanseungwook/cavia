import arguments
from train_huh import run
import os
from utils import set_log, set_seed, Logger
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Set log
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    logger  = Logger(log, tb_writer, args.log_name, args.log_interval)

    # Set seed
    set_seed(args.seed)

    # Start train
    run(args, logger)
