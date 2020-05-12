import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='CAVIA (Regression experiments)')

    # Algorithm
    parser.add_argument(
        '--task', type=str, choices=["mixture"], default='mixture', 
        help="Problem to solve")
    parser.add_argument(
        '--tasks-per-metaupdate', type=int, default=50, 
        help="Number of tasks per meta-update")
    parser.add_argument(
        '--n-sample', type=int, default=10, 
        help='data points in task set')
    parser.add_argument(
        '--lr-inner', type=float, default=0.03,
        help='inner-loop learning rate (task-specific)')
    parser.add_argument(
        '--lr-meta', type=float, default=0.001, 
        help='outer-loop learning rate')
    parser.add_argument(
        '--n-inner', type=int, default=3, 
        help='number of inner-loop updates (during training)')
    parser.add_argument(
        '--n-context-params', type=int, default=2, 
        help='Number of context parameters (added at first layer)')
    parser.add_argument(
        '--n-context-models', type=int, default=2, 
        help='Numbers of context models (i.e., lower-level modules other than the highest module)')
    parser.add_argument(
        '--architecture', type=int, nargs='+', default=[1, 40, 40, 1], 
        help="Architecture of neural network")
    parser.add_argument(
        '--model-type', type=str, choices=["CAVIA"], default="CAVIA", 
        help='model type to use as the base model')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Seed for reproducibility")
    parser.add_argument(
        '--prefix', type=str, default="",
        help="Prefix for logging")
    parser.add_argument(
        "--vis-mode", action="store_true",                                                
        help="If True, perform visualization based on saved iteration")

    args = parser.parse_args()

    # Use GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set args log name
    args.log_name = \
        "n_inner::%s_n_sample::%s_lr_inner::%s_n_context_models::%s_prefix::%s" % (
            args.n_inner, args.n_sample, args.lr_inner, args.n_context_models, args.prefix)

    return args
