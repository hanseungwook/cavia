import torch
from torch.nn.utils import clip_grad_value_
import torch.nn as nn

from hierarchical_task import custom_collate

from pdb import set_trace
# import higher 

##########################################################################################################################################   
# optimize
##########################################################################################################################################   
## optimize parameter for a given 'task'

def optimize(param_all, model_forward, save_cktp_fnc, 
            task_pkg, optim_args,  
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
    def initialize(len_task):     ## Initialize param & optim
        param = param_all[level]
        if param is None:
            return  None
        else:
            if callable(param): #      # use regular optim: for outer-loop
                assert not reset #level == top_level
                optim = optimizer(param(), lr=lr)   # outer-loop: regular optim
                if optimizer_state_dict is not None:
                    optim.load_state_dict(optimizer_state_dict)
                return optim
            else:  # use manual optim
                assert reset
                n_ctx = n_contexts[level]
                param_all[level] = make_zero_ctx(n_ctx, len_task)   #  torch.zeros_like(param_all[level], requires_grad=True, device=device) 
                return  None 

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

    pars, tasks, names = task_pkg

    if collate_tasks: 
        if not isinstance(tasks, list):
            pars = [pars]
            tasks = [tasks]
            names = [names]
        len_task = len(tasks)

        train_loader = [t.load('train') for t in tasks]
        test_loader  = [t.load('test') for t in tasks]
    else:
        len_task = 1 #None

    optim = initialize(len_task)  
    i = iter0
    loss = None


    while True:
        if collate_tasks:
            train_subtasks, train_batch = collect_subtasks(train_loader, names)
            if i >= max_iter or (i > 0 and not i%test_interval):  # Terminal or intermediate (for logging every test_interval) 
                test_subtasks, test_batch = collect_subtasks(test_loader, names)
                test_loss, test_output = model_forward(test_subtasks, sample_type = 'test', iter_num = i, batch = test_batch)
            
                save_cktp_fnc(level, i, test_loss, loss, optim)

                if i >= max_iter:         # Terminate! 
                    return test_loss, test_output 

            i += 1  
            loss, output = model_forward(train_subtasks, sample_type = 'train', iter_num = i, batch = train_batch)    # Loss to be optimized
            update_step()

            # if level<2:
            #     print(param_all[level]*1e4)
        else:
    # while True:
            for train_subtasks in tasks.load('train').loader:     # task_list = sampled mini-batch
                # for _ in range(for_iter):          # Seungwook: for_iter replicates cavia’s implementation where they use the same mini-batch for the inner loop steps
                    
                    if i >= max_iter or (i > 0 and not i%test_interval):  # Terminal or intermediate (for logging every test_interval) 
                        for test_subtasks in tasks.load('test').loader:           # Run Test-loss
                            test_loss, test_output = model_forward(test_subtasks, sample_type = 'test', iter_num = i)
                            break
                        
                        save_cktp_fnc(level, i, test_loss, loss, optim)

                        if i >= max_iter:         # Terminate! 
                            return test_loss, test_output 

                    i += 1  
                    loss, output = model_forward(train_subtasks, sample_type = 'train', iter_num = i)    # Loss to be optimized
                    update_step()

                    # if level<2:
                    #     print(param_all[level]*1e4)


############

def clip_grad_value_fnc(grad, grad_clip):
    for g in grad: #filter(lambda p: p.grad is not None, parameters):
        g.data.clamp_(min=-grad_clip, max=grad_clip)

#################
# collect_subtasks
def collect_subtasks(loader_list, names_prev):

    args_all = [ [], [], [] ] # pars, tasks, names
    
    for (loader, name_prev) in zip(loader_list, names_prev):
        for arg_all, arg in zip(args_all, loader.get_next()):
            if isinstance(arg, tuple):   # tuples of tasks / names
                if isinstance(arg[0],str):  # names
                    arg= [name_prev + '_' + a for a in arg]
                arg_all += list(arg)
            else:
                arg_all.append(arg)  # list of input/param tensors

    for i, arg_ in enumerate(args_all):
        if isinstance(arg_[0], torch.Tensor):
            args_all[i] = torch.stack(arg_,dim=0)

    return args_all, len(arg)   #pars, tasks, names



def make_zero_ctx(n, len_task = None, device = 'cpu'):
    # if DOUBLE_precision:
    #     return torch.zeros(1,n, requires_grad=True).double()
    if len_task is None:
        return torch.zeros(n, requires_grad=True, device=device)
    else:
        return torch.zeros(len_task, n, requires_grad=True, device=device)            