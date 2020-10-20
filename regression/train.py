from rl_utils import make_env
from model import get_model_type
from hierarchical import Hierarchical_Model, get_hierarchical_task


def get_base_model(args, logger):
    model_type = get_model_type(args.model_type, is_rl=True)
    logger.log[args.log_name].info("Selecting base_model: {}".format(model_type))

    # Overwrite last layer of the architecture according to the action space of the environment
    # Note that we put a default env and task only to get the action space of the environment
    env = make_env(args=args)()
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    args.architecture[0] = input_dim
    args.architecture[-1] = action_dim
    env.close()

    model = model_type(n_arch=args.architecture, n_contexts=args.n_contexts, device=args.device).to(args.device)
    logger.log[args.log_name].info("Model: {}".format(model))

    return model


def make_batch_dict(n_trains, n_tests, n_valids):
    return [
        {'train': n_train, 'test': n_test, 'valid': n_valid} 
        for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids)]


def run(args, logger):
    k_batch_dict = make_batch_dict(args.k_batch_train, args.k_batch_test, args.k_batch_valid)
    n_batch_dict = make_batch_dict(args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    task = get_hierarchical_task(
        task_func_list=["MiniGrid-Empty-5x5-v0", "MiniGrid-Unlock-v0"], 
        k_batch_dict=k_batch_dict, 
        n_batch_dict=n_batch_dict,
        is_rl=True)
    # from task.mixture2 import task_func_list
    # task = get_hierarchical_task(
    #     task_func_list=task_func_list, 
    #     k_batch_dict=k_batch_dict, 
    #     n_batch_dict=n_batch_dict,
    #     is_rl=False)

    base_model = get_base_model(args, logger)
    model = Hierarchical_Model(
        decoder_model=base_model, 
        encoders=None, 
        is_rl=True, 
        args=args, 
        task=task)
    model(task, reset=False)  # TODO Pass TRPO instead of Adam
