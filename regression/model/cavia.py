import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Cavia(nn.Module):
    def __init__(self, n_arch, args):
        super(Cavia, self).__init__()
        self.args = args

        self.fc_layers = nn.ModuleList()
        for i_layer in range(len(n_arch) - 1):
            self.fc_layers.append(nn.Linear(n_arch[i_layer], n_arch[i_layer + 1]))

        self.optimizer = optim.Adam(self.parameters(), args.lr_meta)

    def reset_context(self):
        context = torch.zeros(self.args.n_context_params).to(self.args.device)
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

        # Perform prediction
        for i_layer in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i_layer](x))
        y = self.fc_layers[-1](x)

        return y

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
