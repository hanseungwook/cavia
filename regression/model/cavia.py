import torch
import torch.nn as nn
import torch.nn.functional as F


class Cavia(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """
    def __init__(self, n_arch, n_context, device):
        super(Cavia, self).__init__()

        self.device = device

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for k in range(len(n_arch) - 1):
            self.fc_layers.append(nn.Linear(n_arch[k], n_arch[k + 1]))

        # Context parameters 
        # NOTE that these are *not* registered parameters of the model
        self.n_context = n_context

    def reset_context(self):
        context = torch.zeros(self.n_context).to(self.device)
        context.requires_grad = True
        return context

    def forward(self, x, lower_context, higher_context=None):
        # Concatenate input with context parameters
        lower_context = lower_context.expand(x.shape[0], -1)
        if higher_context is None:
            x = torch.cat((x, lower_context), dim=1)
        else:
            higher_context = higher_context.expand(x.shape[0], -1)
            x = torch.cat((x, lower_context, higher_context), dim=1)

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y
