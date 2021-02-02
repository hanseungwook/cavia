import os
import torch
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model.models_huh import get_model_type, get_encoder_type
from hierarchical import Hierarchical_Model,  get_hierarchical_task
from utils import get_vis_fn, print_args

from pdb import set_trace
import IPython

from task.LQR import Combine_NN_ENV        

print_logger_name = True
image_flag = False #True

####################################################
    
def get_base_model(args):
    LQR_flag = (args.task[0] == 'LQR')
    if LQR_flag: 
        return Combine_NN_ENV(x_dim=2) 
    else:
        MODEL_TYPE = get_model_type(args.model_type)
        n_contexts=args.n_contexts    #   n_context=sum(args.n_contexts)
        return MODEL_TYPE(n_arch=args.architecture, n_contexts=args.n_contexts, device=args.device).to(args.device)

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


def load_model(args, model): 
    if args.load_model:
        print('Loading model')
        checkpoint = torch.load('./model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        
    return model, num_test

def get_num_test(args):
#     To Fix: Clean this up.. or provide comment: Huh
#     if args.test_interval == 0:
#         args.test_interval = args.n_iters[-1]
#     num_test = args.n_iters[-1] // args.test_interval 
#     args.n_iters[-1] = args.test_interval   # To fix: This is bad: changing outer-loop n_iter without any warning. 
    num_test = 1
    return num_test
    
###############################################################


def run(args, loggers, test_loggers):
    
    num_test = get_num_test(args)
    
    base_model      = get_base_model(args) 
    encoder_models  = get_encoder_model(args.encoders, args)                   # adaptation model: None == MAML
    model   = Hierarchical_Model(base_model, args.n_contexts, args.n_iters, args.for_iters, args.lrs, encoder_models, loggers, test_loggers, args.data_parallel)
    
    task = get_task(args)

    save_dir = get_save_dir(args.log_name)    
    if args.viz:  # should this be needed? Utilize model run below directly 
        vis_save_fn = get_vis_fn(args.task)

    print('start model training')
    for i in range(num_test):
        test_loss, outputs = model(task, optimizer=Adam, reset=False, return_outputs=True) #outerloop = True)   # grad_clip = args.clip ) #TODO: gradient clipping?
        print('outer-loop idx', i, 'test loss', test_loss.item())
        
        if args.viz:
            print('Saving reconstruction')
            vis_save_fn(outputs, save_dir, i*args.test_interval)

        save_model(save_dir, model)  
    
            
    return test_loss.item() 


#########################################################


def get_task(args):
    def make_batch_dict(n_trains, n_tests, n_valids):
        return [   {'train': n_train, 'test': n_test, 'valid': n_valid} 
                   for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids) ]
    k_batch_dict = make_batch_dict(args.k_batch_train, args.k_batch_test, args.k_batch_valid)
    n_batch_dict = make_batch_dict(args.n_batch_train, args.n_batch_test, args.n_batch_valid)
    return get_hierarchical_task(args.task, args.classes, k_batch_dict, n_batch_dict)
    
def get_loggers(logger_maker, levels):
    def get_logger_list(log_type):
        logger_list = []
        for i in range(levels):
            no_print=(i<levels-1)
            logger_list.append(logger_maker(additional_name=log_type+str(i), no_print=no_print))
            if print_logger_name:
                print('logger_name=', log_type+'_lv'+str(i), 'no_print=', no_print)
        return logger_list

    return get_logger_list(log_type='train'), get_logger_list(log_type='test')

###################################################

def get_save_dir(log_name):
    save_dir = os.path.join('model_save', log_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_model(save_dir, model):
    file_name = 'model.pth'
    print('Saving model')
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_dir, file_name))
            
