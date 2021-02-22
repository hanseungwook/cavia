import os
import torch
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model.models_huh import get_model_type, get_encoder_type
from hierarchical import Hierarchical_Model  #,  get_hierarchical_task
from dataset import Hierarchical_Task #, get_hierarchical_task
from task.make_tasks_new import get_task_dict 
from utils import get_vis_fn #, print_args

from pdb import set_trace
import IPython

from task.LQR import Combine_NN_ENV        

print_logger_name = True
image_flag = False #True

####################################################
    
def get_base_model(hparams):
    task_name = hparams.task #hparams.task[0]
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


def get_task(hparams):
    
    def make_batch_dict(n_trains, n_tests, n_valids):
        return [   {'train': n_train, 'test': n_test, 'valid': n_valid} 
                   for n_train, n_test, n_valid in zip(n_trains, n_tests, n_valids) ]
    
    k_batch_dict = make_batch_dict(hparams.k_batch_train, hparams.k_batch_test, hparams.k_batch_valid)  # Total-batch
    n_batch_dict = make_batch_dict(hparams.n_batch_train, hparams.n_batch_test, hparams.n_batch_valid)  # mini-batch
    
    batch_dict=(k_batch_dict, n_batch_dict)

    task_func = get_task_dict[hparams.task]  #     task_name = hparams.task 


    return task_func, batch_dict

#     task = Hierarchical_Task(task_func, idx=0, batch_dict=(k_batch_dict, n_batch_dict))
#     return [task]

#########################################################


def get_num_test(hparams):
##     Seungwook's comment: This is dividing up the total outer loop # of iterations by the test interval and at each test interval, creating a reconstruction/visaulization.
##     To be Fixed: Cleaned up.. 

    if hparams.test_interval == 0:
        hparams.test_interval = hparams.n_iters[-1]
    num_test = hparams.n_iters[-1] // hparams.test_interval 
    hparams.n_iters[-1] = hparams.test_interval   # To fix: This is bad: changing outer-loop n_iter without any warning. 
    num_test = 1
    return num_test



###############################################################


def run(hparams, logger): #loggers, test_loggers):

    hparams, levels = check_hparam_default(hparams)
    
    num_test = get_num_test(hparams)
    save_dir = get_save_dir(hparams)    
    
    base_model      = get_base_model(hparams) 
    encoder_models  = get_encoder_model(hparams.encoders, hparams)                   # adaptation model: None == MAML
    model   = Hierarchical_Model(levels, base_model, encoder_models, logger, 
                                 hparams.n_contexts, hparams.n_iters, hparams.for_iters, hparams.lrs, 
                                 hparams.loss_logging_levels, hparams.ctx_logging_levels, 
                                 hparams.higher_flag, hparams.data_parallel)
    
    if hparams.v_num is not None: #if hparams.load_model:
        model = load_model(model, save_dir, hparams.log_name, hparams.v_num)
    task_func, batch_dict = get_task(hparams)
    supertask = Hierarchical_Task(task_func, batch_dict=batch_dict, idx=0)
    
    print('start model training')
    
    for i in range(num_test):
        test_loss, outputs = model([supertask], optimizer=Adam, reset=False, return_outputs=True) # grad_clip = hparams.clip ) #TODO: gradient clipping?
        print('outer-loop idx', i, 'test loss', test_loss.item())
        
        if hparams.viz:  # should this be needed? Utilize model run below directly 
            print('Saving reconstruction')
            vis_save_fn = get_vis_fn(hparams.task)
            vis_save_fn(outputs, save_dir, i*hparams.test_interval)

        save_model(model, save_dir)  
        logger.save()
    
    return test_loss.item() 


###################
def check_hparam_default(hparams):
    ## Default copy replacing None
    
    levels   = len(hparams.n_contexts) + 1 #len(decoder_model.parameters_all) #- 1

    hparams.for_iters     = hparams.for_iters or [1]*levels
    hparams.lrs           = hparams.lrs or [0.01]*levels
    
    hparams.k_batch_train = hparams.k_batch_train or [None]*levels
    hparams.n_batch_train = hparams.n_batch_train or [1]*levels  #hparams.k_batch_train
    
    hparams.k_batch_test  = hparams.k_batch_test  or hparams.k_batch_train
    hparams.k_batch_valid = hparams.k_batch_valid or hparams.k_batch_train
    hparams.n_batch_test  = hparams.n_batch_test  or hparams.n_batch_train
    hparams.n_batch_valid = hparams.n_batch_valid or hparams.n_batch_train
    
    
    for name in ['lrs', 'n_iters', 'for_iters', 'k_batch_train', 'k_batch_test', 'k_batch_valid', 'n_batch_train', 'n_batch_test', 'n_batch_valid']:
        assert len(getattr(hparams,name)) == levels, "wrong level "+name
        
    return hparams, levels

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
