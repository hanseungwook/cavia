import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='CAVIA (Regression experiments)')

    parser.add_argument(
        '--task', type=str, choices=["sine", "mixture"], 
        default='mixture', help="Problem to solve")
    # parser.add_argument(
    #     '--tasks-per-metaupdate', type=int, 
    #     default=25, help="Number of tasks per meta-update")
    parser.add_argument(
        '--architecture', type=int, nargs='+', 
        default=[1, 40, 40, 1], help="Architecture of neural network")
    parser.add_argument(
        '--first_order', action='store_true', default=False, 
        help='run first-order version')
    parser.add_argument(
        '--model-type', type=str, choices=["CAVIA", "ADDITIVE", "MULTIPLICATIVE", "ADD_MULTIPLICATIVE"], 
        default="CAVIA", help='model type: CAVIA, ADDITIVE, MULTIPLICATIVE, ADD_MULTIPLICATIVE"')
    parser.add_argument(
        '--seed', type=int, default=42)
    parser.add_argument(
        '--prefix', type=str, default="",
        help="Prefix for logging")
    parser.add_argument(
        '--log-name', type=str, default="",
        help="Logging name")

    parser.add_argument(
        '--lrs', type=float, nargs='+', default=[0.05, 0.05, 0.001],
        help="lr for inner-loop, midloop, outerloop")
    parser.add_argument(
        '--max-iters', type=int, nargs='+', default=[2, 2, 1000], 
        help="optim_iter for inner-loop, midloop, outerloop")
    parser.add_argument(
        '--k_batch_train', type=int, nargs='+', default=[100, 2, 2],
        help="number of datapoints, tasks, super-tasks")
    parser.add_argument(
        '--k_batch_test', type=int, nargs='+', default=[100, 2, 2], 
        help="number of datapoints, tasks, super-tasks")
    parser.add_argument(
        '--k_batch_valid', type=int, nargs='+', default=[100, 2, 2],
        help="number of datapoints, tasks, super-tasks")
    parser.add_argument(
        '--n_batch_train', type=int, nargs='+', default=[100, 2, 2], 
        help="number of datapoints, tasks, super-tasks")
    parser.add_argument(
        '--n_batch_test', type=int, nargs='+', default=[100, 2, 2],
        help="number of datapoints, tasks, super-tasks")
    parser.add_argument(
        '--n_batch_valid', type=int, nargs='+', default=[100, 2, 2], 
        help="number of datapoints, tasks, super-tasks")
    parser.add_argument(
        '--n_contexts', type=int, nargs='+', default=[2, 1],
        help="number of context variables: phi0, phi1")
    parser.add_argument(
        '--encoders', type=int, nargs='+', default=[None, None, None],
        help="task encoder-models for model-based Meta-learning. Optimization-based if None (MAML)")

    parser.add_argument('--log_interval', type=int, default=100)

    # Arguments for reinforcement learning settings
    parser.add_argument(
        '--ep-max-timestep', type=int, default=20, 
        help="Episode horizon")

    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set args log name
    args.log_name = "prefix::%s" % (args.prefix)

    return args
