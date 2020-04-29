import torch.optim as optim
import torch.nn.functional as F

from model.models_huh import get_model_type


### asserts following variables to be implemented as lists: 
# model.ctx[level]   
# args.n_iter[level]
# args.lr[level]
# args.n_batch[level]  = n_batch_train, n_batch_test, n_batch_valid

### Each 'task' object is assumed to have built-in sample() methods, which returns a 'list of subtasks':
# lv 2: task = supertask-set,     subtasks  = supertasks
# lv 1: task = supertask,         subtasks  = tasks (functions)
# lv 0: task = task (function),   subtasks  = (input, output) data-points


def get_model(args):
    n_arch = args.architecture
    n_context = args.n_context_params
    if args.model_type == "CAVIA":
        n_arch[0] += n_context * args.n_context_models
        
    MODEL_TYPE = get_model_type(args.model_type)
    model = MODEL_TYPE( n_arch=n_arch, n_context=n_context, device=args.device).to(args.device)
    return model


def get_task_family(args):
    pass    # Needs a new implementation for tasks with sample() methods.

def run(args, log, tb_writer):
    # Set tasks
    model = get_model(args)
    top_level_task = get_task_family(args)   # returns a list of 1 element: [super-super-task]
    optimize(args, model, args.top_level, top_level_task, log, tb_writer)

    # vis_pca(higher_contexts, task_family, iteration, args)      ## Visualize result


##############################################################################
# recursive functions. See pseudo-code below.  

def evaluate(args, model, level, tasks): 
    if level == 0:
        inputs, targets = tasks
        return F.mse_loss(model(inputs), targets)
    else:           
        loss = 0                     # going 1 level lower
        for task in tasks:
            optimize        (args, model, level-1, task.sample('train'))      # optimize model.ctx[level]    # subtasks_train      # sample dataset (subtasks)
            loss += evaluate(args, model, level-1, task.sample('test'))                                      # subtasks_test       # sample dataset (subtasks)
        return loss / float(len(tasks))  
 


def optimize(args, model, level, tasks, log = None, tb_writer = None):      # optimize model.parameter[level]  for a given 'task'
    if level == args.top_level:  
        optimizer = optim.Adam(model.parameters(), args.lr[level])
    else: 
        optimizer = manual_optim([model.ctx[level]], args.lr[level])     # replacing optim.SGD .  check for memory leak

    for i_iter in range(args.n_iter[level]): 
        loss = evaluate(args, model, level, tasks)  

        if level == args.top_level:  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ------------ logging ------------
           log[args.log_name].info("At iteration {}, meta-loss: {:.3f}".format(
                iteration, meta_loss.detach().cpu().numpy()))
            tb_writer.add_scalar("Meta loss:", meta_loss.detach().cpu().numpy(), iteration)

        else:
            model.ctx[level].grad = torch.autograd.grad(loss, model.ctx[level], create_graph=True)[0]     # create_graph= not args.first_order)[0]
            optimizer.step()             #  check for memory leak
            # if memory_leak:
            # grad = torch.autograd.grad(loss, model.ctx[level], create_graph=True)[0]                 # create_graph= not args.first_order)[0]
            # model.ctx[level] = model.ctx[level] - args.lr[level] * grad


#########################
# Manual optim :  replace optim.SGD  for memory leak problem
# manual_optim = optim.SGD

class manual_optim():
    def __init__(self, param_list, lr):
        self.param_list = param_list
        self.lr = lr

    def zero_grad(self):
        for par in self.param_list:
            par.grad = torch.zeros(par.data.shape, device=par.device)

    def step(self):
        for par in self.param_list:
            par.data = par.data - self.lr * par.grad
            # par.data -= self.lr * par.grad
            # par = par - self.lr * par.grad
            # par -= par.grad * self.lr   # this is bad...


#########################################
# Pseudo-code
 
# def evaluate(level = 0, tasks = datapoints = [datapoint_i])       : fit each datapoint  (for given function)
# return loss = F.mse_loss(datapoints)  

# def optimize(level = 0):
#     for loop:
#         loss = evaluate(level = 0)
#         backward_step()


# def evaluate(level = 1,  tasks = fncs = [fnc_i])                : fit each function  (for given function_type e.g. 'sinusoid')
# for fnc_i in tasks:
#     sample train_datapoints_i, test_datapoints_i
#     phi0_i = optimize(level = 0,  task = train_datapoints_i)     # optimize phi0_i   
#     loss_i = evaluate(level = 0,  task = test_datapoints_i)      # using    phi0_i  
# return sum(loss_i)

# def optimize(level = 1, task):
#     for loop:
#         loss = evaluate(level = 1, task)
#         backward_step()

# def evaluate(level = 2, tasks = fnc_types = [fnc_type_i])       : fit each function_type  (e.g. 'sinusoid')
# for fnc_type_i in tasks:
#     sample train_fncs_i, test_datapoints_i
#     phi1_i = optimize(level = 1,  task = train_fncs_i)       # optimize phi1_i   
#     loss_i = evaluate(level = 1,  task = test_fncs_i)        # using    phi1_i  
# return sum(loss_i)

# def optimize(level = 2,  task = ALL_TASK):      # ALL_TASK = [function_types]
#     for loop:
#         loss = evaluate(level = 2, task)  # 1 task
#         backward_step()

# theta = optimize(level = 2,  tasks = [function_types])       # optimize theta = phi2 




