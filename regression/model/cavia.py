import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CAVIA(nn.Module):
    def __init__(self, args, logger):
        super(CAVIA, self).__init__()

        self.args = args
        self.logger = logger
        self.module_list = nn.ModuleList()
        for i_layer in range(len(args.network_arch) - 1):
            self.module_list.append(nn.Linear(args.network_arch[i_layer], args.network_arch[i_layer + 1]))

    def forward(self, x, ctx):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Concatenate input with context
        ctx = torch.cat(ctx, dim=1)
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        # Get categorical distribution
        for i, layer in enumerate(self.module_list):
            x = layer(x)
            x = F.relu(x) if i < len(self.module_list) - 1 else x
        return Categorical(logits=x)
