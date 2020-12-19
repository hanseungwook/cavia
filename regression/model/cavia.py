import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class CAVIA(nn.Module):
    def __init__(self, args, logger):
        super(CAVIA, self).__init__()

        self.args = args
        self.logger = logger
        self.module_list = nn.ModuleList()

        self.fc1 = nn.Linear(args.network_arch[0], args.network_arch[1])
        self.fc2 = nn.Linear(args.network_arch[1], args.network_arch[2])
        if args.is_continuous_action:
            self.fc3_mu = nn.Linear(args.network_arch[2], args.network_arch[3])
            self.fc3_sigma = nn.Parameter(torch.Tensor(args.network_arch[-1]))
            self.min_log_std = math.log(1e-6)
        else:
            self.fc3 = nn.Linear(args.network_arch[2], args.network_arch[3])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, ctx):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Concatenate input with context
        ctx = torch.cat(ctx, dim=1)
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        # Process through neural network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Get distribution
        if self.args.is_continuous_action:
            mu = self.fc3_mu(x)
            scale = torch.exp(torch.clamp(self.fc3_sigma, min=self.min_log_std))
            return Normal(loc=mu, scale=scale)
        else:
            x = self.fc3(x)
            return Categorical(logits=x)
