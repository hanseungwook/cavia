import os
import torch
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model.models_huh import get_model_type, get_encoder_type
from hierarchical import Hierarchical_Model,  get_hierarchical_task
from utils import get_vis_fn #, print_args

from pdb import set_trace
import IPython

from task.LQR import Combine_NN_ENV        

print_logger_name = True
image_flag = False #True

####################################################
    
def get_base_model(hparams):
    LQR_flag = (hparams.task[0] in ['LQR', 'LQR_lv2','LQR_lv1'])
    if LQR_flag: 
        print(hparams.task[0])
        if hparams.task[0] == 'LQR_lv2':
            return Combine_NN_ENV(x_dim=2, levels = 2) 
        elif hparams.task[0] == 'LQR_lv1':
            return Combine_NN_ENV(x_dim=2, levels = 1) 
        else:
            error()
    else:
        MODEL_TYPE = get_model_type(hparams.model_type)
        n_contexts=hparams.n_contexts    #   n_context=sum(hparams.n_contexts)
        return MODEL_TYPE(n_arch=hparams.architecture, n_contexts=hparams.n_contexts, device=hparams.device).to(hparams.device)

def get_encoder_model(encoder_types, hparams):
    encoders = []
    for encoder_type in encoder_types:
        if encoder_type is None:
            encoders.append(None)
        else:
            ENCODER_TYPE = get_encoder_type(encoder_type)   # Check if n_hidden is the task embedding dimension in Encoder_Core().
            encoder_model = ENCODER_TYPE( input_dim=hparams.input_dim, n_hidden=hparams.n_hidden, tree_hidden_dim=hparams.tree_hidden_dim, 
                                cluster_layer_0=hparams.cluster_layer_0, cluster_layer_1=hparams.cluster_layer_1).to(hparams.device)
            encoders.append(encoder_model)
    return encoders



def get_task(hparams):
    def make_batch_dict(n_trains, n_tests, n_valids):
        return [   {'train': n_train, 'test': n_test, 'valid': n_valid} 
                   for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids) ]
    k_batch_dict = make_batch_dict(hparams.k_batch_train, hparams.k_batch_test, hparams.k_batch_valid)
    n_batch_dict = make_batch_dict(hparams.n_batch_train, hparams.n_batch_test, hparams.n_batch_valid)
    return get_hierarchical_task(hparams.task, hparams.classes, k_batch_dict, n_batch_dict)
    
    

#########################################################

# def get_loggers(logger_maker, levels):
#     def get_logger_list(log_type):
#         logger_list = []
#         for i in range(levels):
#             no_print=(i<levels-1)
#             logger_name = log_type+'_lv'+str(i)
#             logger_list.append(logger_maker(additional_name=logger_name, no_print=no_print))
#             if print_logger_name:
#                 print('logger_name=', logger_name, 'no_print=', no_print)
#         return logger_list
#     return get_logger_list(log_type='train'), get_logger_list(log_type='test')


def get_num_test(hparams):
##     Seungwook's comment: This is dividing up the total outer loop # of iterations by the test interval and at each test interval, creating a reconstruction/visaulization.
##     To be Fixed: Cleaned up.. 

#     if hparams.test_interval == 0:
#         hparams.test_interval = hparams.n_iters[-1]
#     num_test = hparams.n_iters[-1] // hparams.test_interval 
#     hparams.n_iters[-1] = hparams.test_interval   # To fix: This is bad: changing outer-loop n_iter without any warning. 
    num_test = 1
    return num_test

###################################################

def get_save_dir(log_name):
    save_dir = os.path.join('model_save', log_name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    return save_dir

def save_model(save_dir, model):
    file_name = 'model.pth'
    print('Saving model')
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_dir, file_name))
            

def load_model(hparams, model): 
    if hparams.load_model:
        print('Loading model')
        checkpoint = torch.load('./model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        
    return model, num_test



###############################################################


def run(hparams, logger): #loggers, test_loggers):

    logger.log_hyperparams(hparams)
    logger.save() 

    num_test = get_num_test(hparams)
    save_dir = get_save_dir(hparams.log_name)    
    
    base_model      = get_base_model(hparams) 
    encoder_models  = get_encoder_model(hparams.encoders, hparams)                   # adaptation model: None == MAML
    model   = Hierarchical_Model(base_model, hparams.n_contexts, hparams.n_iters, hparams.for_iters, hparams.lrs, encoder_models, logger, hparams.ctx_logging_levels, hparams.higher_flag, hparams.data_parallel)
    
    task = get_task(hparams)
    
    print('start model training')
    
    for i in range(num_test):
        test_loss, outputs = model(task, optimizer=Adam, reset=False, return_outputs=True) # grad_clip = hparams.clip ) #TODO: gradient clipping?
        print('outer-loop idx', i, 'test loss', test_loss.item())
        
        if hparams.viz:  # should this be needed? Utilize model run below directly 
            print('Saving reconstruction')
            vis_save_fn = get_vis_fn(hparams.task)
            vis_save_fn(outputs, save_dir, i*hparams.test_interval)

        save_model(save_dir, model)  
    
    return test_loss.item() 
