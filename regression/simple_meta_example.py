
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from functools import partial
from torch.optim import Adam, SGD

from pdb import set_trace

max_iter_list = [10, 2, 2]

class BaseModel(nn.Module):

    def __init__(self, *layers):
        super().__init__()
        parameter_all_levels = [list(layers[level].parameters()) for level in range(3)]

        self.layers = nn.ModuleList(list(layers))        
        self.parameter_this_level = parameter_all_levels[0]
        self.parameter_next_levels = parameter_all_levels[1:]
        self.level = len(parameter_all_levels)-1
        self.nonlin = nn.ReLU()
        self.loss_fnc = nn.MSELoss()

    def forward(self, data_batch):
        x, targets = data_batch
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.nonlin(x)  if i<len(self.layers)-1 else x
        return self.loss_fnc(x, targets), outputs  

################

def optimize(model, data, level, max_iter, optimizer):       # optimize parameter for a given 'task'
    params = model.parameter_this_level
    optim = optimizer(params,  lr = 0.1) 
    for _ in range(max_iter):
        loss = model(data) [0] 
        optim_backward_step(optim, loss)
        if level==0:
            print(loss.item())

def optim_backward_step(optim, loss):
    optim.zero_grad()
    if hasattr(optim, 'backward'):   # manual_optim case  # cannot call loss.backward() in inner-loops
        optim.backward(loss)
    else:
        loss.backward()
        optim.step()   

class manual_optim():
    def __init__(self, param_list, lr):
        self.param_list = param_list
        self.lr = lr

    def zero_grad(self):
        for par in self.param_list:
            par.grad = torch.zeros_like(par.data)               # par.grad = torch.zeros(par.data.shape, device=par.device)   

    def backward(self, loss):
        grad_list = torch.autograd.grad(loss, self.param_list, create_graph=True)
        for par, grad in zip(self.param_list, grad_list):
            par = par - self.lr * grad

################

def make_hierarhical_model(model):   
    while len(model.parameter_next_levels) > 0:
        model = Hierarchical_Model(model) 
    return model


class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, submodel): 
        super().__init__()
        self.submodel  = submodel 
        self.level = len(submodel.parameter_next_levels)-1
        self.parameter_this_level = submodel.parameter_next_levels[0]
        self.parameter_next_levels = submodel.parameter_next_levels[1:]
        self.max_iter = max_iter_list[self.level]

    def forward(self, data, optimizer = manual_optim): 
        optimize(self.submodel, data, self.level, self.max_iter, optimizer=optimizer)      
        test_loss, outputs = self.submodel(data)    
        return test_loss, outputs 


basemodel = BaseModel(nn.Linear(1,5),nn.Linear(5,5),nn.Linear(5,1))
model = make_hierarhical_model(basemodel)
data = torch.randn(10,1), torch.randn(10,1)
model(data, Adam)
# set_trace()
