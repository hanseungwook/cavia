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

    parser.add_argument('--lrs',           type=float, nargs='+', default=[0.05, 0.05, 0.001])     # lr  for inner-loop, midloop, outerloop
    parser.add_argument('--n_iters',       type=int, nargs='+', default=[3, 2, 1000])             # optim_iter for inner-loop, midloop, outerloop
    parser.add_argument('--k_batch_train', type=int, nargs='+', default=[100, 25, 2])              # number of datapoints, tasks, super-tasks 
    parser.add_argument('--k_batch_test',  type=int, nargs='+', default=[100, 25, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--k_batch_valid', type=int, nargs='+', default=[100, 25, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_train', type=int, nargs='+', default=[30, 15, 2])              # number of datapoints, tasks, super-tasks 
    parser.add_argument('--n_batch_test',  type=int, nargs='+', default=[30, 15, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_valid', type=int, nargs='+', default=[30, 15, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_contexts',    type=int, nargs='+', default=[2, 1])                  # number of context variables: phi0, phi1 
    parser.add_argument('--encoders',      type=int, nargs='+', default=[None, None])            # task encoder-models for model-based Meta-learning. Optimization-based if None (MAML) 
    parser.add_argument('--log_interval',  type=int, default=100)

    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set args log name
    args.log_name = \
        "test::" 
        # "iter::%s_lr::%sn_batch::%s_n_ctx::%s_prefix::%s" % (
        #     args.n_iters, args.lrs, args.n_batch_train, len(args.n_contexts), args.prefix)
    # args.log_name = \
    #     "n_inner::%s_k-meta-train::%s_lr_inner::%s_n_context_models::%s_prefix::%s" % (
    #         args.n_inner, args.k_meta_train, args.lr_inner, args.n_context_models, args.prefix)

    return args
