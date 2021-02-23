
import torch
import torch.nn as nn
from functools import partial
from torch.optim import Adam, SGD
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import numpy as np
# from copy import deepcopy
from pdb import set_trace

# import higher


# max_iter_list = [2, 2, 3]
max_iter_list = [3,3, 20]
# max_iter_list = [1,1, 2]

grad_clip_value = 100  #1000

n_ctx = [2,3]                # n_ctx = [0,0] 
n_input = 1 + sum(n_ctx)

debug_level =  3 #2 
DOUBLE_precision = True #False #

lr = 0.01

layers = nn.Linear(n_input,5), nn.Linear(5,5), nn.Linear(5,1)

if DOUBLE_precision:
    data = torch.randn(10,1).double(), torch.randn(10,1).double()
else:
    data = torch.randn(10,1), torch.randn(10,1)

def make_ctx(n):
#     if DOUBLE_precision:
#         return torch.zeros(1,n, requires_grad=True).double()
#     else:
        return torch.zeros(1,n, requires_grad=True)

################

class BaseModel(nn.Module):

    def __init__(self, n_ctx, *layers):
        super().__init__()

        self.layers = nn.ModuleList(list(layers))        
        self.parameters_all = [make_ctx(n) for n in n_ctx] + [self.layers.parameters]
        self.nonlin = nn.ReLU()
        self.loss_fnc = nn.MSELoss()
        
    def model_forward(self, x, ctx):
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.nonlin(x)  if i<len(self.layers)-1 else x
        return x

    def forward(self, data_batch):
        inputs, targets = data_batch
        ctx = torch.cat(self.parameters_all[:-1], dim=1)            # print(ctx)        # print(self.parameters_all[:-1])
        outputs = self.model_forward(inputs, ctx)
        loss = self.loss_fnc(outputs, targets) #, outputs  
        return loss

##############################

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, submodel): 
        super().__init__()
        self.submodel  = submodel 
        self.level_max = len(submodel.parameters_all)  #2

    def forward(self, data, level = None, optimizer = None): 
        if level is None:
            level = self.level_max

        if level == 0:
            return self.submodel(data)
        else:
            reset_flag = level<self.level_max:
            optimize(self, data, level-1, max_iter_list[level-1], optimizer=optimizer, reset_flag)
            test_loss = self(data, level-1)    

        if level == self.level_max - 1: #in [2,3]:
            print('level', level, test_loss.item())

        return test_loss #, outputs 
    
    
# model(theta, ctx1, ctx0, x) = y

# model2 = partial(model, theta) lv2 
# model1 = partial(model2, ctx1) lv1
# model0 = partial(model1, ctx0) lv0
# y = model0(x)

# def evalulate0(model0, task0):
#     x, tar = task0
#     out = model0(x)
#     loss = loss_fnc(out, tar)
#     return loss

# def optimize0(model0, task0_train):
#     for _ in range(max_iter):
#         loss = evalulate0(model0, task0_train)
#         loss.backward(ctx0)
#         optim.step()

# def evalulate1(model1, task1):
#     initialize_ctx0()  # ctx0 = 0
#     model0 = partial(model1, ctx0)
#     task0_train = task1.get_lower('train')
#     optimize0(model0, task0_train)
    
#     task0_test = task1.get_lower('test')
#     loss = evalulate0(model0, task0_test)

#     return loss


# def Hierarchical_Model(nn.Module):  
#     def __init__(self, base_model):
#         self.base_model = base_model
    
#     def evaluate(self, task_):
#         lower_tasks = task_.lower()
#         if lower_tasks is not None: #  level == 0:
#             loss = get_average_loss(self, lower_tasks)
#         else:
#             loss = self.base_model(task_)
#         return loss
    
#     def forward(task):
#         optimize(self, task('train'))  #  inner-loop optim
#         test_loss = self.evaluate(task('test'))
#         return test_loss
    
    
# def get_average_loss(model, tasks):
#     loss = []
#     for t_ in tasks:
#         l = model(t_)
#         loss.append(l)
#     return  torch.stack(loss).mean() 


################

def optimize(model, data, level, max_iter, optimizer, reset_flag):      
    param_all = model.submodel.parameters_all
    if reset_flag:
        param_all[level] = torch.zeros_like(param_all[level], requires_grad= True)
    else:
        optim = optimizer(param_all[level](), lr=lr)

    for _ in range(max_iter):
        loss = model(data, level) # [0] 

        if reset:
            grad = torch.autograd.grad(loss, param_all[level], create_graph = True)
#             for g in grad: # filter(lambda p: p.grad is not None, parameters):
#                 g.data.clamp_(min=-grad_clip_value, max=grad_clip_value)
            param_all[level] = param_all[level] - lr * grad[0]
            
        else:
            optim.zero_grad()
            loss.backward()
            # if level == debug_level: # == 2:
            #     grad_list = list(par.grad for par in optim.param_groups[0]['params'])
            #     debug(model, data, level, list(param_all[level]()), grad_list)
            optim.step()  


################


# def check_grad(fnc, input, eps = 1e-6):
#     grad_num = torch.zeros_like(input)
#     for i in range(input.numel()):
#         idx = np.unravel_index(i, input.shape)
#         in_ = input.clone().detach();  in_.requires_grad = True;       in_[idx] += eps;           
#         loss1 = fnc(in_)
#         in_ = input.clone().detach();  in_.requires_grad = True;       in_[idx] -= eps;           
#         loss2 = fnc(in_)
#         grad_num[idx] = (loss1 - loss2)/2/eps
#     return grad_num

# def eval_model_weight(model, minibatch, level, input = None):  # for outerloop
#     model_ = model
#     if input is not None:
#         vector_to_parameters(input, model_.parameters())
#     return model_(minibatch, level) 

# def debug(model, data, level, params, grad_list):
#     fnc = partial(eval_model_weight, model, data, level)
#     finite_grad = check_grad(fnc, parameters_to_vector(params))
#     numeric_grad = parameters_to_vector(grad_list)
#     print('level', level, ', grad error: ', (finite_grad - numeric_grad).norm().item(), ', fd: ', (finite_grad).norm().item(), ', num: ', (numeric_grad).norm().item())

##################

def main():
    basemodel = BaseModel(n_ctx, *layers)
    model = Hierarchical_Model(basemodel) 

    if DOUBLE_precision:
        model.double()

    model(data, optimizer=Adam, reset=False)


if __name__ == '__main__':
    main()
#     hparams = get_args()
#     main(hparams)