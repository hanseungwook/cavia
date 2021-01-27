import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Normal


class BaseModelRL(nn.Module):
    def __init__(self, n_arch, n_contexts, nonlin, loss_fnc, device, is_continuous_action, FC_module):
        super().__init__()

        self.module_list = nn.ModuleList()
        for i in range(len(n_arch) - 1):
            self.module_list.append(FC_module(n_arch[i], n_arch[i + 1]))    # Fully connected layers
        self.parameters_all = [self.make_ctx(n) for n in n_contexts] + [self.module_list.parameters]
        self.device = device if device is not None else 'cpu'
        self.n_contexts = n_contexts
        self.nonlin = nonlin
        self.loss_fnc = loss_fnc
        self.n_arch = n_arch
        if is_continuous_action:
            self.sigma = nn.Parameter(torch.Tensor(output_size))
            self.sigma.data.fill_(math.log(init_std))

    def forward(self, x):
        raise NotImplementedError()

    def reset_context(self):
        self.parameters_all = [self.make_ctx(n) for n in self.n_contexts] + [self.module_list.parameters]

    @staticmethod
    def make_ctx(n):
        return torch.zeros(1, n, requires_grad=True)


class CaviaRL(BaseModelRL):
    def __init__(self, n_arch, n_contexts, is_continuous_action, nonlin=nn.ReLU(), loss_fnc=None, device=None):
        n_arch[0] += sum(n_contexts)  # add n_context to n_input 
        super(CaviaRL, self).__init__(n_arch, n_contexts, nonlin, loss_fnc, device, is_continuous_action, FC_module=nn.Linear)

    def forward(self, x, layers=None, ctx_=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Concatenate input with context
        if ctx_ is None:
            ctx = self.parameters_all[:-1]
        else:
            ctx = ctx_

        if len(x.shape) == 3:
            ctx = torch.cat(ctx, dim=1)
            x = torch.cat((x, ctx.expand(x.shape[0], x.shape[1], -1)), dim=-1)
        else:
            ctx = torch.cat(ctx, dim=1)
            x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        # Get categorical distribution
        layers_ = self.module_list if layers is None else layers
    
        for i, layer in enumerate(layers_):
            x = layer(x)
            x = self.nonlin(x) if i < len(layers_) - 1 else x
            if i == len(layers_) - 1:
                if is_continuous_action:
                    scale = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        
        if is_continuous_action:
            return Normal(loc=x, scale=scale)
        else:
            return Categorical(logits=x)
