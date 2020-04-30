import arguments
import train_huh
import os
from utils import set_log, set_seed
# from tensorboardX import SummaryWriter


if __name__ == '__main__':
    # Load arguments
    args = arguments.parse_args()

    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Set log
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))

    # Set seed
    set_seed(args.seed)

    # Start train
    train_huh.run(args, log, tb_writer)
