import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='CAVIA (Regression experiments)')

    # Arguments for algorithm
    parser.add_argument(
        '--task', type=str, choices=["empty", "unlock", "mixture", "ant"], 
        help="Problem to solve")
    parser.add_argument(
        '--architecture', type=int, nargs='+', default=[1, 40, 40, 1], 
        help="Architecture of neural network")
    parser.add_argument(
        '--model-type', type=str, choices=["CAVIA", "ADDITIVE", "MULTIPLICATIVE", "ADD_MULTIPLICATIVE"], default="CAVIA", 
        help='model type: CAVIA, ADDITIVE, MULTIPLICATIVE, ADD_MULTIPLICATIVE"')
    parser.add_argument(
        '--lrs', type=float, nargs='+', default=[0.02, 0.02, 0.001],
        help="lr for inner-loop, midloop, outerloop")
    parser.add_argument(
        '--max-iters', type=int, nargs='+', default=[2, 2, 10000], 
        help="optim_iter for inner-loop, midloop, outerloop")
    parser.add_argument(
        '--batch', type=int, nargs='+', default=[25, 5, 2], 
        help="number of trajectories, tasks (e.g., goal locations), super-tasks (e.g., empty)")
    parser.add_argument(
        '--n_contexts', type=int, nargs='+', default=[5, 5],
        help="number of context variables: phi0, phi1")
    parser.add_argument(
        '--is_continuous_action', action='store_true', default=False,
        help='turn on continuous action')

    # Arguments for reinforcement learning settings
    parser.add_argument(
        '--ep-max-timestep', type=int, default=20, 
        help="Episode horizon")

    # Arguments for misc
    parser.add_argument(
        '--log-name', type=str, default="",
        help="Logging name")
    parser.add_argument(
        '--seed', type=int, default=42, 
        help="Seed for reproducibility")
    parser.add_argument(
        '--prefix', type=str, default="",
        help="Prefix for logging")

    args = parser.parse_args()

    # use the GPU if available
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cpu")

    # Set args log name
    args.log_name = \
        "task::%s_lrs::%s_max_iters::%s_n_contexts::%s_batch::%s_prefix::%s" % (
            args.task, args.lrs, args.max_iters, args.n_contexts, args.batch, args.prefix)

    return args
