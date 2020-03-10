import os
import pickle
import arguments
import cavia
import maml


if __name__ == '__main__':

    args = arguments.parse_args()

    if args.maml:
        logger = maml.run(args, log_interval=100, rerun=True)
        with open(args.logger_save_file, 'wb') as f:
            pickle.dump(logger, f)
    else:
        logger = cavia.run(args, log_interval=100, rerun=True)
        with open(args.logger_save_file, 'wb') as f:
            pickle.dump(logger, f)
