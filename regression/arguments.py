import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='CAVIA (Regression experiments)')

    parser.add_argument(
        '--task', type=str, choices=["sine", "mixture"], 
        default='mixture', help="Problem to solve")
    parser.add_argument(
        '--tasks-per-metaupdate', type=int, 
        default=25, help="Number of tasks per meta-update")
    parser.add_argument(
        '--k-meta-train', type=int, default=10, 
        help='data points in task training set (during meta training, inner loop)')
    parser.add_argument(
        '--k-meta-test', type=int, default=10, 
        help='data points in task test set (during meta training, outer loop)')
    parser.add_argument(
        '--lr-inner', type=float, default=0.01,
        help='inner-loop learning rate (task-specific)')
    parser.add_argument(
        '--lr-meta', type=float, default=0.001, 
        help='outer-loop learning rate')
    parser.add_argument(
        '--n-inner', type=int, default=1, 
        help='number of inner-loop updates (during training)')
    parser.add_argument(
        '--n-context-params', type=int, default=5, 
        help='Number of context parameters (added at first layer)')
    parser.add_argument(
        '--n-context-models', type=int, default=2, 
        help='Numbers of context models (i.e., lower-level modules other than the highest module)')
    parser.add_argument(
        '--architecture', type=int, nargs='+', 
        default=[1, 40, 40, 1], help="Architecture of neural network")
    parser.add_argument(
        '--num_hidden_layers', type=int, nargs='+', default=[40, 40])
    parser.add_argument(
        '--first_order', action='store_true', default=False, 
        help='run first-order version')
    parser.add_argument(
        '--model-type', type=str, choices=["CAVIA"], 
        default="CAVIA", help='model type: ACTIVE or CAVIA')

    parser.add_argument(
        '--seed', type=int, default=42)
    parser.add_argument(
        '--log-name', type=str, default="",
        help="Logging name")

    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set args log name
    args.log_name = \
        "n_inner::%s_lr_inner::%s_n_context_models::%s_" % (args.n_inner, args.lr_inner, args.n_context_models)

    return args
