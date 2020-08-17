import higher
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torch.optim import Adam, SGD
from torch.nn.utils import parameters_to_vector, vector_to_parameters


max_iter_list = [3, 3, 20]
n_ctx = [2, 3]
n_input = 1 + sum(n_ctx)
debug_level = 3
DOUBLE_precision = True
lr = 0.01
layers = nn.Linear(n_input, 5), nn.Linear(5, 5), nn.Linear(5, 1)

if DOUBLE_precision:
    data = torch.randn(10, 1).double(), torch.randn(10, 1).double()
else:
    data = torch.randn(10, 1), torch.randn(10, 1)


def make_ctx(n):
    if DOUBLE_precision:
        return torch.zeros(1, n, requires_grad=True).double()
    else:
        return torch.zeros(1, n, requires_grad=True)


class BaseModel(nn.Module):
    def __init__(self, n_ctx, *layers):
        super().__init__()

        self.layers = nn.ModuleList(list(layers))        
        self.parameters_all = [make_ctx(n) for n in n_ctx] + [self.layers.parameters]
        self.nonlin = nn.ReLU()
        self.loss_fnc = nn.MSELoss()

    def forward(self, data_batch):
        x, targets = data_batch
        ctx = torch.cat(self.parameters_all[:-1], dim=1)
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.nonlin(x) if i < len(self.layers) - 1 else x
        outputs = x
        return self.loss_fnc(outputs, targets)


class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self, submodel): 
        super().__init__()
        self.submodel = submodel 
        self.level_max = len(submodel.parameters_all)

    def forward(self, data, level=None, optimizer=None, reset=True): 
        if level is None:
            level = self.level_max

        if level == 0:
            return self.submodel(data)
        else:
            optimize(self, data, level - 1, max_iter_list[level - 1], optimizer=optimizer, reset=reset)
            test_loss = self(data, level - 1)

        if level == self.level_max - 1:
            print('level', level, test_loss.item())

        return test_loss


def optimize(model, data, level, max_iter, optimizer, reset):      
    param_all = model.submodel.parameters_all
    if reset:
        param_all[level] = torch.zeros_like(param_all[level], requires_grad=True)
        optimizer = SGD
        optim = optimizer([param_all[level]], lr=lr)
        optim = higher.get_diff_optim(optim, [param_all[level]])
    else:
        optim = optimizer(param_all[level](), lr=lr)

    for _ in range(max_iter):
        loss = model(data, level)

        if reset:
            param_all[level], = optim.step(loss, params=[param_all[level]])
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()  


def check_grad(fnc, input, eps=1e-6):
    grad_num = torch.zeros_like(input)

    for i in range(input.numel()):
        idx = np.unravel_index(i, input.shape)
        in_ = input.clone().detach()
        in_.requires_grad = True
        in_[idx] += eps
        loss1 = fnc(in_)
        in_ = input.clone().detach()
        in_.requires_grad = True
        in_[idx] -= eps
        loss2 = fnc(in_)
        grad_num[idx] = (loss1 - loss2) / 2 / eps

    return grad_num


def eval_model_weight(model, minibatch, level, input=None):
    model_ = model
    if input is not None:
        vector_to_parameters(input, model_.parameters())
    return model_(minibatch, level) 


def debug(model, data, level, params, grad_list):
    fnc = partial(eval_model_weight, model, data, level)
    finite_grad = check_grad(fnc, parameters_to_vector(params))
    numeric_grad = parameters_to_vector(grad_list)
    print(
        'level', level, 
        ', grad error: ', (finite_grad - numeric_grad).norm().item(), 
        ', fd: ', (finite_grad).norm().item(), 
        ', num: ', (numeric_grad).norm().item())


basemodel = BaseModel(n_ctx, *layers)
model = Hierarchical_Model(basemodel) 

if DOUBLE_precision:
    model.double()

model(data, optimizer=Adam, reset=False)
