import torch
from torch.nn.utils import clip_grad_value_
import torch.nn as nn

# import higher 
print_optimize_level_iter = False #True #
print_optimize_level_over = False


##########################################################################################################################################   
# optimize
##########################################################################################################################################   

def optimize(model, dataloader, level, 
            lr, max_iter, for_iter, optimizer, reset, 
            status, # current_status, 
            run_test, test_interval, 
            device, Higher_flag, 
            log_ctx_flag):  #log_loss_flag, 
    ## optimize parameter for a given 'task'
    grad_clip_value = 100 #1000
    # print(level)
    
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
                        g.data.clamp_(min=-grad_clip_value, max=grad_clip_value)
                    param_all[level] = param_all[level] - lr * grad[0]

            else: # use regular optim: for outer-loop
                optim.zero_grad()
                loss.backward()
                clip_grad_value_(param_all[level](), clip_value=grad_clip_value) #20)
                optim.step()  
                
    ######################################
    # main code

    param_all, optim = initialize()  
    cur_iter = 0
    loss = None
    # status = status # + current_status
    while True:
        for i, task_batch in enumerate(dataloader):
            for _ in range(for_iter):          # Seungwook: for_iter is to replicate cavia’s implementation where they use the same mini-batch for the inner loop steps
                if cur_iter >= max_iter:       # Terminate! 
                    # if print_optimize_level_over and loss is not None:
                    #     print('optimize '+ status, 'loss_final', loss.item())
                    return None   # param_all[level]    # for log_ctx

                loss, output = model.forward(level, task_batch, status = status, #current_status = current_status, 
                                                                iter_num = cur_iter)    # Loss to be optimized
                update_step()

                if log_ctx_flag and reset:
                    model.log_ctx(param_all[level], status + '_lv'+str(level), cur_iter)  #  log the adapted ctx for the level
                # logging(loss.item(), param_all[level], cur_iter)


                # Run Test-loss
                if not (cur_iter % test_interval) and cur_iter <= max_iter - test_interval:
                    test_loss, test_outputs = run_test(iter_num = cur_iter)  # get test_loss 
                    if level>0:
                        print('level',level, 'cur_iter', cur_iter, 'test loss', test_loss.item())

                cur_iter += 1  