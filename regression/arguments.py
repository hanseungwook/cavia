import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Toy example experiment')

    # Arguments for algorithm
    parser.add_argument(
        "--network_arch", type=int, nargs='+', default=[-1, 100, 100, -1],
        help="Architecture of neural network")
    parser.add_argument(
        "--learning-rates", type=float, nargs='+', default=[0.05, 0.05, 0.005],
        help="lr for inner-loop, midloop, outerloop")
    parser.add_argument(
        "--max-iterations", type=int, nargs='+', default=[2, 2, 10000],
        help="optim_iter for inner-loop, midloop, outerloop")
    parser.add_argument(
        "--batch", type=int, nargs='+', default=[20, 20, 2],
        help="number of trajectories, tasks (e.g., goal locations), super-tasks (e.g., empty)")
    parser.add_argument(
        "--n-contexts", type=int, nargs='+', default=[1, 1],
        help="number of context variables: phi0, phi1")
    parser.add_argument(
        "--is-hierarchical", action="store_true",
        help="If True, perform hierarchical meta-learning")
    parser.add_argument(
        "--discount", type=float, default=0.95,
        help="Discount factor")
    parser.add_argument(
        "--tau", type=float, default=0.95,
        help="Lambda factor in GAE")

    # Arguments for env
    parser.add_argument(
        "--task", type=str, nargs='+', default=[],
        help="Problem to solve")
    parser.add_argument(
        "--ep-max-timestep", type=int, default=25,
        help="Episode horizon")

    # Arguments for misc
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Seed for reproducibility")
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for logging")

    args = parser.parse_args()

    # Set args log name
    args.log_name = \
        "task::%s_learning_rates::%s_max_iterations::%s_batch::%s_args.n_contexts::%s_is_hierarchical::%s_prefix::%s" % (
            args.task, "-".join(str(item) for item in args.learning_rates),
            "-".join(str(item) for item in args.max_iterations), "-".join(str(item) for item in args.batch),
            "-".join(str(item) for item in args.n_contexts), args.is_hierarchical, args.prefix)

    return args
