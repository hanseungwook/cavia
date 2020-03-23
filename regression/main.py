import os
import pickle
import arguments
import train
import maml


if __name__ == '__main__':

    args = arguments.parse_args()

    if args.maml:
        print('Running maml')
        logger = maml.run(args, log_interval=args.log_interval, rerun=True)
        with open(args.logger_save_file, 'wb') as f:
            pickle.dump(logger, f)
    elif args.onehot:
        print('Running 1 hot encoder version')
        logger = train.run_no_inner(args, log_interval=args.log_interval, rerun=True)
        with open(args.logger_save_file, 'wb') as f:
            pickle.dump(logger, f)
    else:
        logger = train.run(args, log_interval=args.log_interval, rerun=True)
        with open(args.logger_save_file, 'wb') as f:
            pickle.dump(logger, f)
