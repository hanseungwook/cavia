import os
import torch
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import numpy as np
from model.models_huh import get_model_type, get_encoder_type
from hierarchical import Hierarchical_Model,  get_hierarchical_task   # make_hierarhical_model,
from utils import get_vis_fn, save_img_recon

from pdb import set_trace
import IPython

def get_base_model(args):
    MODEL_TYPE = get_model_type(args.model_type)
    # model = MODEL_TYPE( n_arch=args.architecture, n_context=sum(args.n_contexts), device=args.device).to(args.device)
    model = MODEL_TYPE(n_arch=args.architecture, n_contexts=args.n_contexts, device=args.device).to(args.device)
    return model

def get_encoder_model(encoder_types, args):
    encoders = []
    for encoder_type in encoder_types:
        if encoder_type is None:
            encoders.append(None)
        else:
            ENCODER_TYPE = get_encoder_type(encoder_type)   # Check if n_hidden is the task embedding dimension in Encoder_Core().
            encoder_model = ENCODER_TYPE( input_dim=args.input_dim, n_hidden=args.n_hidden, tree_hidden_dim=args.tree_hidden_dim, 
                                cluster_layer_0=args.cluster_layer_0, cluster_layer_1=args.cluster_layer_1).to(args.device)
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
    task = get_hierarchical_task(args.task, k_batch_dict, n_batch_dict)    ### FIX THIS : task_func_list
    
    base_model      = get_base_model(args)
    encoder_models  = get_encoder_model(args.encoders, args)                   # adaptation model: None == MAML
    loggers         = [logger_maker(additional_name='0', no_print=True), logger_maker(additional_name='1', no_print=True), logger_maker(additional_name='2', no_print=False)]
    test_loggers         = [logger_maker(additional_name='test_0', no_print=True), logger_maker(additional_name='test_1', no_print=True), logger_maker(additional_name='test_2', no_print=True)]
    if args.test_interval == 0:
        args.test_interval = args.n_iters[-1]
    num_test = args.n_iters[-1] // args.test_interval
    args.n_iters[-1] = args.test_interval

    model   = Hierarchical_Model(base_model, args.n_contexts, args.n_iters, args.for_iters, args.lrs, encoder_models, loggers, test_loggers)
    # set_trace()
    if args.load_model:
        print('Loading model')
        checkpoint = torch.load('./model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        
    if args.viz:
        vis_fn = get_vis_fn(args.task)
        vis_fn(model, task)
        
    else:
        for i in range(num_test):
            test_loss, outputs = model(task, optimizer=Adam, reset=False, return_outputs=True) #outerloop = True)   # grad_clip = args.clip ) #TODO: gradient clipping?
            print('test loss', test_loss)

            save_dir = args.log_name
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            print('Saving reconstruction')
            img_pred = outputs.view(28, 28, 1).detach().cpu().numpy()

            # Forcing all predictions beyond image value range into (0, 1)
            img_pred = np.clip(img_pred, 0, 1)
            plt.imshow(img_pred)
            plt.savefig(os.path.join(save_dir, 'recon_img_itr{}.png'.format(i*args.test_interval)))
            
            print('Saving model')
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_dir, 'model.pth'))
    
    # return test_loss, logger 
