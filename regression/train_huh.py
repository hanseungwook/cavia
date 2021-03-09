import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model.models_huh import get_model_type, get_encoder_type
# from hierarchical_eval import Hierarchical_Eval  
# from utils import get_vis_fn

import shutil


from pdb import set_trace

from task.LQR import Combine_NN_ENV        

print_logger_name = True
image_flag = False #True



###############################################################

# def get_Hierarchical_Eval(hparams, logger, model_loss = None):

#     base_model, base_loss  = model_loss or get_base_model(hparams) 
#     encoder_models  = get_encoder_model(hparams.encoders, hparams)                   # adaptation model: None == MAML
#     model  = Hierarchical_Eval(hparams, base_model, encoder_models, base_loss, logger)
    
#     if hparams.v_num is not None: # fix load_model ! 
#         model = load_model(model, ckpt_dir, hparams.log_name, hparams.v_num)

#     return model



####################################################
    
def get_base_model(hparams):
    task_name = hparams.task
    LQR_flag = (task_name in ['LQR']) #, 'LQR_lv2','LQR_lv1','LQR_lv0'])
    if LQR_flag: 
        model = Combine_NN_ENV(x_dim=2, levels = 2) 
        base_loss = None
    else:
        MODEL_TYPE = get_model_type(hparams.model_type)
        model = MODEL_TYPE(n_arch=hparams.architecture, n_contexts=hparams.n_contexts, device=hparams.device).to(hparams.device)
        base_loss =  torch.nn.MSELoss()                    # loss function
    return  model,  base_loss

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
        # n_trains, n_tests, n_valids = n_trains+[1], n_tests+[0], n_valids+[0]  # add super-task level
        return [   {'train': n_train, 'test': n_test, 'valid': n_valid} 
                for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids) ]
    
    k_dict = make_batch_dict(hparams.k_train, hparams.k_test, hparams.k_valid)  # Total-batch
    n_dict = make_batch_dict(hparams.n_train, hparams.n_test, hparams.n_valid)  # mini-batch
    return k_dict, n_dict


###################################################

# def get_save_dir(hparams, v_num):
#     save_dir = os.path.join(hparams.save_path, 'logs', hparams.log_name, 'version_' + str(v_num))
#     ckpt_dir = os.path.join(save_dir, 'checkpoints')
#     if not os.path.isdir(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     return save_dir #, ckpt_dir

def save_model(model, dir_):
    file_name = 'model.pth'
    print('Saving model')
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(dir_, file_name))
            

# def load_model(model, ckpt_path): 
#     ckpt = torch.load(ckpt_path)
#     model.load_state_dict(ckpt['model_state_dict'])
#     optimizer_state_dict = ckpt['optimizer_state_dict']
#     epoch = ckpt['epoch']
#     best_loss = ckpt['test_loss']

#     del ckpt      
#     return model, optimizer_state_dict, epoch, best_loss


######
def load_model(model, ckpt_dir, v_num):

    ckpt_file = get_latest_ckpt(ckpt_dir)
    if v_num is not None:
        if ckpt_file is None: 
            raise error()
        else:
            ckpt = torch.load(ckpt_file)
            model.load_state_dict(ckpt['model_state_dict'])
            return model, ckpt['optimizer_state_dict'], ckpt['epoch'], ckpt['test_loss']
    else:
        return model, None, 0, None  # model, optimizer_state_dict, epoch0, best_loss

###############################################


import glob

def get_ckpt_dir(log_dir):
    ckpt_dir = os.path.join(log_dir, 'checkpoints') 
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)            
    return ckpt_dir

def get_latest_ckpt(ckpt_path):
    # ckpt_path = os.path.join(save_dir, 'checkpoints', '*')
    filename_list = glob.glob(os.path.join(ckpt_path, '*'))
    name_sorted = sorted(
        filename_list, 
        # key = lambda f: (int(f.split('-')[0].split('=')[1])), #, int(f.split('-')[1].split('=')[1].split('.')[0])),
        key = lambda f: int(f.split('-')[0].split('=')[1].split('.')[0]),
        reverse = True
    )
    return name_sorted[0] if len(name_sorted)>0 else None

#############################

def save_cktp(ckpt_dir, filename, epoch, test_loss, train_loss = None, model_state_dict = None, optimizer_state_dict = None):
    torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict, #optimizer.state_dict() if optimizer is not None else None,
                }, os.path.join(ckpt_dir,filename))


def _del_file(filepath):
    if filepath is None:
        pass
    else:
        dirpath = os.path.dirname(filepath)
        # make paths
        os.makedirs(dirpath, exist_ok=True)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
