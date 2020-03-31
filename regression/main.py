import pickle
import arguments
import train


if __name__ == '__main__':
    args = arguments.parse_args()

    logger = train.run(args, log_interval=args.log_interval, rerun=True)
    with open(args.logger_save_file, 'wb') as f:
        pickle.dump(logger, f)
