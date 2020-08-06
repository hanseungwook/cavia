from functools import partial
import torch
import numpy as np
import pdb

### Functions for finite difference method for calculating gradient

def debug_top(model, params, minibatch, analy_grad):
    # print(list(params))
    # print(analy_grad)
    fnc = partial(eval_model_weight, model, minibatch)
    print_grad_err(fnc, params, analy_grad = analy_grad, level = 2)

def debug_lower(model, params, minibatch, loss, ctx, level):
    grad = torch.autograd.grad(loss, params, create_graph=True)[0]             #     # print('level', self.level, 'debug')
    fnc = partial(eval_submodel_weight, model, minibatch, [])
    print_grad_err(fnc, params[0], analy_grad = grad, level = level)   # params[0] = ctx



def print_grad_err(fnc, params, analy_grad = None, level = 0):
    params = torch.nn.utils.parameters_to_vector(params)
        
    finite_grad = check_grad(fnc, params, eps=1e-8, analytic_grad=False)

    # if analy_grad is None:
    # analy_grad = torch.cat([p.grad.view(-1) for p in params])

    print('level', level, ', grad error: ', (finite_grad - analy_grad).norm().detach().numpy())


def check_grad(fnc, input, eps = 1e-6, analytic_grad = False):
    # print(input)
    grad_num = torch.zeros_like(input)

    for i in range(input.numel()):
        idx = np.unravel_index(i, input.shape)

        in_ = input.clone().detach();  in_.requires_grad = True;       in_[idx] += eps;           loss1, _ = fnc(in_)
        in_ = input.clone().detach();  in_.requires_grad = True;       in_[idx] -= eps;           loss2, _ = fnc(in_)
        grad_num[idx] = (loss1 - loss2)/2/eps

    # if analytic_grad:
    #     in_ = input.clone().detach()
    #     in_.requires_grad = True
    #     assert(in_.requires_grad)
    #     loss = fnc(input.clone().detach());         loss.backward(); 
    #     return grad_num, in_.grad
    # else:
    return grad_num


def eval_model_weight(model, minibatch, input):  # for outerloop
    torch.nn.utils.vector_to_parameters(input, model.parameters())
    return model(minibatch) 

def eval_submodel_weight(submodel, minibatch, ctx_high, ctx):
    return submodel(minibatch, ctx_high + [ctx]) 
