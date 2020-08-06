from torch.optim import Adam, SGD
from model.models_huh import get_model_type, get_encoder_type
from task.mixture2 import task_func_list                             ### FIX THIS 
from hierarchical import make_hierarhical_model, get_hierarhical_task 



def get_base_model(args):
    MODEL_TYPE = get_model_type(args.model_type)
    model = MODEL_TYPE( n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
    # model = MODEL_TYPE( n_arch=args.architecture, n_context=args.n_contexts, device=args.device).to(args.device)
    return model

def get_encoder_model(encoder_types, args):
    encoders = []
    for encoder_type in encoder_types:
        if encoder_type is None:
            encoders.append(None)
        else:
            ENCODER_TYPE = get_encoder_type(encoder_type)   # Check if n_hidden is the task embedding dimension in Encoder_Core().
            encoder_model = ENCODER_TYPE( input_dim=args.input_dim, n_hidden=args.n_hidden, tree_hidden_dim=args.tree_hidden_dim, 
                                cluster_layer_0=args.cluster_layer_0, cluster_layer_1=args.cluster_layer_1)
            encoders.append(encoder_model)
    return encoders

def make_batch_dict(n_trains, n_tests, n_valids):
    return [
            {'train': n_train, 'test': n_test, 'valid': n_valid} 
            for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids)
            ]


def run(args, logger_maker):
    k_batch_dict = make_batch_dict(args.k_batch_train, args.k_batch_test, args.k_batch_valid)
    n_batch_dict = make_batch_dict(args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    task = get_hierarhical_task(task_func_list, k_batch_dict, n_batch_dict)    ### FIX THIS : task_func_list
    
    base_model      = get_base_model(args)
    encoder_models  = get_encoder_model(args.encoders, args)                   # adaptation model: None == MAML
    loggers         = [None, None, logger_maker()]                             # For 2-level problems.  FIX THIS
    model           = make_hierarhical_model(base_model, args.n_contexts, args.n_iters, args.lrs, encoder_models, loggers)

    test_loss = model( task, ctx_high = [], optimizer = Adam, outerloop = True)   # grad_clip = args.clip )
    # return test_loss, logger 
