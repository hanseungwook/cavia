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
            n_contexts,
            collate_tasks,
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
                # param_all[level] = torch.zeros_like(param_all[level], requires_grad=True, device=device)   # Reset
                param_all[level] = make_zero_ctx(n_contexts[level])
                optim = None
            else:         # use regular optim: for outer-loop
                optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim
                if optimizer_state_dict is not None:
                    optim.load_state_dict(optimizer_state_dict)
            return param_all, optim

    def expand_ctx(batch):     ## Initialize param & optim
        param_all[level] = param_all[level].squeeze(0).expand([batch] + list(param_all[level].shape))

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

    if collate_tasks: 
        if isinstance(task, list):
            expand_ctx(len(task))
        else:
            task = [task]

        train_loader = [t.load('train') for t in task]
        test_loader = [t.load('test') for t in task]

    while True:
        if collate_tasks:
            train_subtasks = collect_subtasks(train_loader, level)
            if i >= max_iter or (i > 0 and not i%test_interval):  # Terminal or intermediate (for logging every test_interval) 
                test_subtasks = collect_subtasks(test_loader, level)
                test_loss, test_output = model_forward(test_subtasks, sample_type = 'test', iter_num = i)
            
                save_cktp_fnc(level, i, test_loss, loss, optim)

                if i >= max_iter:         # Terminate! 
                    return test_loss, test_output 

            i += 1  

            loss, output = model_forward(train_subtasks, sample_type = 'train', iter_num = i)    # Loss to be optimized
            update_step()

        else:
    # while True:
            for train_subtasks in task.load('train').loader:     # task_list = sampled mini-batch
                for _ in range(for_iter):          # Seungwook: for_iter replicates cavia’s implementation where they use the same mini-batch for the inner loop steps
                    
                    if i >= max_iter or (i > 0 and not i%test_interval):  # Terminal or intermediate (for logging every test_interval) 
                        for test_subtasks in task.load('test').loader:           # Run Test-loss
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

###

def collect_subtasks(loader_list, level):
    # subtask = []

    pars, tasks, names = [], [], []

    for loader in loader_list:
        batch = loader.get_next()
        if level>0:
            for par_, task_, name_ in batch:
                pars.append(par_); tasks.append(task_); names.append(name_); 
        else:
            par_, task_, name_ = batch
            pars.append(par_); tasks.append(task_); names.append(name_); 

    if isinstance(task_ ,torch.FloatTensor):
        pars = torch.stack(pars, dim=0)
        tasks = torch.stack(tasks, dim=0)        
    return pars, tasks, names



def make_zero_ctx(n, device = 'cpu'):
    # if DOUBLE_precision:
    #     return torch.zeros(1,n, requires_grad=True).double()
    # else:
        # return torch.zeros(1,n, requires_grad=True, device=device)
        return torch.zeros(n, requires_grad=True, device=device)