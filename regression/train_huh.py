import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model.models_huh import get_model_type, get_encoder_type
from hierarchical_eval import Hierarchical_Eval  
# from utils import get_vis_fn

from pdb import set_trace
import IPython

from task.LQR import Combine_NN_ENV        

print_logger_name = True
image_flag = False #True



###############################################################

def get_Hierarchical_Eval(hparams, logger):
    base_model      = get_base_model(hparams) 
    encoder_models  = get_encoder_model(hparams.encoders, hparams)                   # adaptation model: None == MAML
    base_loss       = torch.nn.MSELoss()                    # loss function
    model   = Hierarchical_Eval(hparams, base_model, encoder_models, base_loss, logger)
    
    if hparams.v_num is not None: #if hparams.load_model:
        model = load_model(model, save_dir, hparams.log_name, hparams.v_num)

    return model



####################################################
    
def get_base_model(hparams):
    task_name = hparams.task
    LQR_flag = (task_name in ['LQR', 'LQR_lv2','LQR_lv1','LQR_lv0'])
    if LQR_flag: 
        print(task_name)
        if task_name == 'LQR_lv2':
            return Combine_NN_ENV(x_dim=2, levels = 2) 
        elif task_name == 'LQR_lv1':
            return Combine_NN_ENV(x_dim=2, levels = 1) 
        else:
            error()
    else:
        MODEL_TYPE = get_model_type(hparams.model_type)
        n_contexts=hparams.n_contexts    #  sum(hparams.n_contexts)
        return MODEL_TYPE(n_arch=hparams.architecture, n_contexts=n_contexts, device=hparams.device).to(hparams.device)

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


def get_batch_dict(hparams):
    def make_batch_dict(n_trains, n_tests, n_valids): 
        n_trains, n_tests, n_valids = n_trains+[1], n_tests+[0], n_valids+[0]  # add super-task level
        return [   {'train': n_train, 'test': n_test, 'valid': n_valid} 
                for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids) ]
    
    k_dict = make_batch_dict(hparams.k_train, hparams.k_test, hparams.k_valid)  # Total-batch
    n_dict = make_batch_dict(hparams.n_train, hparams.n_test, hparams.n_valid)  # mini-batch
    return (k_dict, n_dict)


###################################################

def get_save_dir(hparams):
    save_dir = os.path.join(hparams.save_path, 'logs', hparams.log_name)
    print(hparams.log_name)
    print(save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    return save_dir

def save_model(model, dir_):
    file_name = 'model.pth'
    print('Saving model')
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(dir_, file_name))
            

def load_model(model, dir_): 
#     dir_ = os.path.join(hparams.save_path, 'model_save', log_name)
    file_name = 'model.pth'
    if hparams.load_model:
        print('Loading model')
        checkpoint = torch.load(os.path.join(dir_, file_name))   #('./model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        
    return model

