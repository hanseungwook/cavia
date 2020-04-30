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
        '--model-type', type=str, choices=["CAVIA"], 
        default="CAVIA", help='model type: CAVIA, ADDITIVE, MULTIPLICATIVE, ADD_MULTIPLICATIVE"')
    parser.add_argument(
        '--seed', type=int, default=42)
    parser.add_argument(
        '--prefix', type=str, default="",
        help="Prefix for logging")
    parser.add_argument(
        '--log-name', type=str, default="",
        help="Logging name")

    parser.add_argument('--lrs',           type=int, nargs='+', default=[0.001, 0.01, 0.01])     # lr         for outerloop, midloop, inner-loop 
    parser.add_argument('--n_iters',       type=int, nargs='+', default=[100, 2, 1])         # optim_iter for outerloop, midloop, inner-loop
    parser.add_argument('--n_contexts',    type=int, nargs='+', default=[5, 5])           # number of context variables: phi0, phi1 
    parser.add_argument('--n_batch_train', type=int, nargs='+', default=[3, 5, 30])   # number of super-tasks, tasks, datapoints
    parser.add_argument('--n_batch_test',  type=int, nargs='+', default=[3, 5, 30])
    parser.add_argument('--n_batch_valid', type=int, nargs='+', default=[3, 5, 30])

    parser.add_argument('--log_interval', type=int, nargs='+', default=100)


    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set args log name
    args.log_name = \
        "iter::%s_lr::%sn_batch::%s_n_ctx::%s_prefix::%s" % (
            args.n_iters, args.lrs, args.n_batch_train, len(args.n_contexts), args.prefix)
    # args.log_name = \
    #     "n_inner::%s_k-meta-train::%s_lr_inner::%s_n_context_models::%s_prefix::%s" % (
    #         args.n_inner, args.k_meta_train, args.lr_inner, args.n_context_models, args.prefix)

    return args
