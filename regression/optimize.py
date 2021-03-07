import torch
from torch.nn.utils import clip_grad_value_
import torch.nn as nn

from pdb import set_trace
# import higher 

##########################################################################################################################################   
# optimize
##########################################################################################################################################   
## optimize parameter for a given 'task'

def optimize(param_all, model_forward, save_cktp_fnc, 
            task, optim_args,  
            optimizer, optimizer_state_dict, 
            reset, 
            device, 
            iter0 = 0,
            Higher_flag = False, 
            grad_clip = 100): 

    level, lr, max_iter, for_iter, test_interval = optim_args

    ################
    def initialize():     ## Initialize param & optim
        if param_all[level] is None:
            return None, None
        else:
            if reset:    # use manual optim
                param_all[level] = torch.zeros_like(param_all[level], requires_grad=True, device=device)   # Reset
                optim = None
            else:         # use regular optim: for outer-loop
                optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim
                if optimizer_state_dict is not None:
                    optim.load_state_dict(optimizer_state_dict)
            return param_all, optim

    def update_step():
        if param_all[level] is None:   
            pass
        else:
            if reset:  # use manual SGD step
                first_order = False
                if param_all[level+1] is None:  # for LQR_.. without network model parameters
                    first_order = True
                    
                grad = torch.autograd.grad(loss, param_all[level], create_graph=not first_order)
                clip_grad_value_fnc(grad, grad_clip)
                param_all[level] = param_all[level] - lr * grad[0]

            else: # use regular optim: for outer-loop
                optim.zero_grad()
                loss.backward()
                clip_grad_value_(param_all[level](), clip_value=grad_clip) #20)
                optim.step()  
                
    ######################################
    # main code

    param_all, optim = initialize()  
    i = iter0
    loss = None

    while True:
        for train_subtasks in task.load('train'):     # task_list = sampled mini-batch
            for _ in range(for_iter):          # Seungwook: for_iter replicates cavia’s implementation where they use the same mini-batch for the inner loop steps
                
                if i >= max_iter or (i > 0 and not i%test_interval):  # Terminal or intermediate (for logging every test_interval) 
                    for test_subtasks in task.load('test'):           # Run Test-loss
                        test_loss, test_output = model_forward(test_subtasks, sample_type = 'test', iter_num = i)
                        break
                    
                    save_cktp_fnc(level, i, test_loss, loss, optim)

                    if i >= max_iter:         # Terminate! 
                        return test_loss, test_output 

                i += 1  

                loss, output = model_forward(train_subtasks, sample_type = 'train', iter_num = i)    # Loss to be optimized
                update_step()


############

def clip_grad_value_fnc(grad, grad_clip):
    for g in grad: #filter(lambda p: p.grad is not None, parameters):
        g.data.clamp_(min=-grad_clip, max=grad_clip)
