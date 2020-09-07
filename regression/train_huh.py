from rl_utils import make_env
from model.models_huh import get_model_type, get_encoder_type
from task.mixture2 import task_func_list  # FIX THIS 
from hierarchical import Hierarchical_Model, get_hierarchical_task


def get_base_model(args):
    model_type = get_model_type(args.model_type, is_rl=True)
    print("Selecting base_model: {}".format(model_type))

    # Change last layer of the architecture according to the action space of the environment
    # Note that we put a dummy env and task only to get the action space of the environment
    env = make_env(args=args)()
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    args.architecture[0] = input_dim
    args.architecture[-1] = action_dim
    env.close()

    model = model_type(n_arch=args.architecture, n_contexts=args.n_contexts, device=args.device).to(args.device)
    print("base_model: {}".format(model))

    return model


def get_encoder_model(encoder_types, args):
    encoders = []
    for encoder_type in encoder_types:
        if encoder_type is None:
            encoders.append(None)
        else:
            # Check if n_hidden is the task embedding dimension in Encoder_Core()
            ENCODER_TYPE = get_encoder_type(encoder_type)
            encoder_model = ENCODER_TYPE(
                input_dim=args.input_dim, n_hidden=args.n_hidden, tree_hidden_dim=args.tree_hidden_dim, 
                cluster_layer_0=args.cluster_layer_0, cluster_layer_1=args.cluster_layer_1)
            encoders.append(encoder_model)
    return encoders


def make_batch_dict(n_trains, n_tests, n_valids):
    return [
        {'train': n_train, 'test': n_test, 'valid': n_valid} 
        for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids)]


def run(args, logger_maker):
    k_batch_dict = make_batch_dict(args.k_batch_train, args.k_batch_test, args.k_batch_valid)
    n_batch_dict = make_batch_dict(args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    task = get_hierarchical_task(task_func_list, k_batch_dict, n_batch_dict)  # FIX THIS : task_func_list
    
    base_model = get_base_model(args)
    loggers = [None, None, logger_maker()]
    model = Hierarchical_Model(base_model, None, loggers, is_rl=True, args=args, task=task)
    test_loss = model(task, reset=False)  # TODO Pass TRPO instead of Adam
    return test_loss
