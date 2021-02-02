import argparse
import torch


def run_argparse(input = None):
    parser = argparse.ArgumentParser(description='CAVIA (Regression experiments)')

    parser.add_argument(
        '--task', type=str, choices=['sine', 'linear', 'quadratic', 'cubic', 'celeba', 'cifar10', 'hier-imagenet', 'celeba_airplane', 'airplane', 'mnist', 'fashion_mnist', 'mnist_fmnist', 'mnist_fmnist_3level'], 
        nargs='+', help="Supertasks to solve")
    parser.add_argument(
        '--classes', type=int, nargs='+', 
        default=[], help="Specified classes of dataset to use (if not specified, all)")
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
        '--viz', action='store_true', default=False, 
        help='Run visualize (with already trained model)')
    parser.add_argument(
        '--load_model', type=str, default='', 
        help='Path to model weights to load')
    parser.add_argument(
        '--data_parallel', action='store_true', default=False, 
        help='Use data parallel for inner model (decoder)')


    parser.add_argument('--lrs',           type=float, nargs='+', default=[0.05, 0.05, 0.001])     # lr  for inner-loop, midloop, outerloop
    parser.add_argument('--n_iters',       type=int, nargs='+', default=[3, 2, 1000])              # max_iter for inner-loop, midloop, outerloop
    parser.add_argument('--for_iters',     type=int, nargs='+', default=[3, 1, 1])                 # number of iterations to use the same data
    parser.add_argument('--k_batch_train', type=int, nargs='+', default=[100, 25, 2])              # number of datapoints, tasks, super-tasks 
    parser.add_argument('--k_batch_test',  type=int, nargs='+', default=[100, 25, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--k_batch_valid', type=int, nargs='+', default=[100, 25, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_train', type=int, nargs='+', default=[30, 15, 2])              # number of datapoints, tasks, super-tasks 
    parser.add_argument('--n_batch_test',  type=int, nargs='+', default=[30, 15, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_batch_valid', type=int, nargs='+', default=[30, 15, 2])              # number of datapoints, tasks, super-tasks
    parser.add_argument('--n_contexts',    type=int, nargs='+', default=[2, 1])                  # number of context variables: phi0, phi1 
    parser.add_argument('--encoders',      type=int, nargs='+', default=[None, None, None])            # task encoder-models for model-based Meta-learning. Optimization-based if None (MAML) 

    parser.add_argument('--log_interval',   type=int, default=100)
    parser.add_argument('--test_interval',  type=int, default=0)
    parser.add_argument('--log_name', nargs='?', type=str, default='test::')
#     parser.add_argument('--log_name',       type=str, nargs='+', default='test::')

    if input is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input)

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set args log name
#     if not args.log_name:
#         args.log_name = \
#             "test::" 
#             # "iter::%s_lr::%sn_batch::%s_n_ctx::%s_prefix::%s" % (
#             #     args.n_iters, args.lrs, args.n_batch_train, len(args.n_contexts), args.prefix)
#     # args.log_name = \
#     #     "n_inner::%s_k-meta-train::%s_lr_inner::%s_n_context_models::%s_prefix::%s" % (
#     #         args.n_inner, args.k_meta_train, args.lr_inner, args.n_context_models, args.prefix)

    return args
