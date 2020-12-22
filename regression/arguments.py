import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CAVIA (Regression experiments)')

    # Arguments for algorithm
    parser.add_argument(
        "--task", type=str, choices=["empty", "unlock", "mixture"],
        help="Problem to solve")
    parser.add_argument(
        "--network_arch", type=int, nargs='+', default=[1, 100, 100, 1],
        help="Architecture of neural network")
    parser.add_argument(
        "--lrs", type=float, nargs='+', default=[0.05, 0.05, 0.005],
        help="lr for inner-loop, midloop, outerloop")
    parser.add_argument(
        "--max-iters", type=int, nargs='+', default=[2, 2, 10000],
        help="optim_iter for inner-loop, midloop, outerloop")
    parser.add_argument(
        "--batch", type=int, nargs='+', default=[20, 20, 2],
        help="number of trajectories, tasks (e.g., goal locations), super-tasks (e.g., empty)")
    parser.add_argument(
        "--n_contexts", type=int, nargs='+', default=[5, 5],
        help="number of context variables: phi0, phi1")
    parser.add_argument(
        "--is-hierarchical-learning", action="store_true",
        help="If True, perform hierarchical meta-learning")

    # Arguments for reinforcement learning settings
    parser.add_argument(
        "--discount", type=float, default=0.95,
        help="Discount factor")
    parser.add_argument(
        "--tau", type=float, default=0.95,
        help="Lambda factor in GAE")
    parser.add_argument(
        "--ep-max-timestep", type=int, default=100,
        help="Episode horizon")

    # Arguments for misc
    parser.add_argument(
        "--log-name", type=str, default="",
        help="Logging name")
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Seed for reproducibility")
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for logging")

    args = parser.parse_args()

    # Set args log name
    args.log_name = \
        "task::%s_lrs::%s_max_iters::%s_batch::%s_args.n_contexts::%s_is_hierarchical_learning::%s_prefix::%s" % (
            args.task, "-".join(str(item) for item in args.lrs), "-".join(str(item) for item in args.max_iters),
            "-".join(str(item) for item in args.batch), "-".join(str(item) for item in args.n_contexts),
            args.is_hierarchical_learning, args.prefix)

    return args
