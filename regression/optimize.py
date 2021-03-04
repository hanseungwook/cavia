import torch
from torch.nn.utils import clip_grad_value_
import torch.nn as nn

# import higher 

##########################################################################################################################################   
# optimize
##########################################################################################################################################   
## optimize parameter for a given 'task'

def optimize(model, dataloader, level, lr, max_iter, for_iter, test_interval, 
            status, status_dict,
            test_eval, 
            optimizer, optimizer_state_dict, 
            reset, 
            device, 
            iter0=0,
            Higher_flag = False, 
            grad_clip = 100): 

    
    ################
    def initialize():     ## Initialize param & optim
        param_all = model.decoder_model.module.parameters_all if isinstance(model.decoder_model, nn.DataParallel) else model.decoder_model.parameters_all

        optim = None
        if param_all[level] is not None:
            if reset:  # use manual optim or higher 
                param_all[level] = torch.zeros_like(param_all[level], requires_grad=True, device=device)   # Reset
                if Higher_flag: # using higher 
                    pass
                    # optim = optimizer([param_all[level]], lr=lr)
                    # optim = higher.get_diff_optim(optim, [param_all[level]]) #, device=x.device) # differentiable optim for inner-loop:
            else: # use regular optim: for outer-loop
                optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim
                if optimizer_state_dict is not None:
                    optim.load_state_dict(optimizer_state_dict)
        return param_all, optim

    def update_step():
        if param_all[level] is not None:      
            if reset:  # use manual SGD or higher 
                if Higher_flag:
                    param_all[level], = optim.step(loss, params=[param_all[level]])    # syntax for diff_optim
                else: # manual SGD step
                    first_order = False
                    if param_all[level+1] is None:  # for LQR_.. without network model parameters
                        first_order = True
                        
                    grad = torch.autograd.grad(loss, param_all[level], create_graph=not first_order)
                    for g in grad: #filter(lambda p: p.grad is not None, parameters):
                        g.data.clamp_(min=-grad_clip, max=grad_clip)
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
        for task_list in dataloader:     # task_list = sampled mini-batch
            for _ in range(for_iter):          # Seungwook: for_iter is to replicate cavia’s implementation where they use the same mini-batch for the inner loop steps
                # Run Test-loss (for logging)
                if not (i % test_interval) or i >= max_iter: # and i <= max_iter - test_interval:
                    test_loss, test_output = test_eval(i) 
                    if level == model.top_level:
                        model.save_cktp(loss, test_loss, i, optim)
                    if i >= max_iter:         # Terminate! 
                        return test_loss, test_output 

                loss, output = model.forward(task_list, sample_type = 'train', level=level, status = status, status_dict = status_dict, iter_num = i)    # Loss to be optimized
                update_step()

                i += 1  