import gym
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from misc.linear_baseline import LinearFeatureBaseline
from torch.distributions import Categorical, Normal
from gym_env import make_env


class CAVIA(nn.Module):
    def __init__(self, args, logger):
        super(CAVIA, self).__init__()

        self.args = args
        self.logger = logger

        # Set network and optmizer
        self.fc1 = nn.Linear(args.network_arch[0], args.network_arch[1])
        self.fc2 = nn.Linear(args.network_arch[1], args.network_arch[2])
        if args.is_continuous_action:
            self.fc3_mu = nn.Linear(args.network_arch[2], args.network_arch[3])
            self.fc3_sigma = nn.Parameter(torch.Tensor(args.network_arch[-1]))
            self.min_log_std = math.log(1e-6)
        else:
            self.fc3 = nn.Linear(args.network_arch[2], args.network_arch[3])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Set baseline
        if args.is_hierarchical:
            input_size = args.network_arch[0] - sum(args.n_contexts)
        else:
            input_size = args.network_arch[0] - args.n_contexts[0]
        self.baseline = LinearFeatureBaseline(input_size)

    def forward(self, x, ctx):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Concatenate input with context
        ctx = torch.cat(ctx, dim=1) if self.args.is_hierarchical else ctx[0]
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        # Process through neural network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Return distribution
        if self.args.is_continuous_action:
            mu = self.fc3_mu(x)
            scale = torch.exp(torch.clamp(self.fc3_sigma, min=self.min_log_std))
            return Normal(loc=mu, scale=scale)
        else:
            x = self.fc3(x)
            return Categorical(logits=x)


def get_base_model(args, logger):
    # Overwrite last layer of the architecture according to the action space of the environment
    # Note that we put a default env and task only to get the action space of the environment
    env = make_env(args=args)()
    input_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.box.Box):
        args.is_continuous_action = True
        action_dim = env.action_space.shape[0]
    else:
        args.is_continuous_action = False
        action_dim = env.action_space.n
    args.network_arch[0] = input_dim
    args.network_arch[-1] = action_dim
    env.close()

    # Overwrite input layer of the architecture with number of context parameters
    if args.is_hierarchical:
        args.network_arch[0] += sum(args.n_contexts)
    else:
        args.network_arch[0] += args.n_contexts[0]

    # Return base_model
    base_model = CAVIA(args, logger)
    logger.log[args.log_name].info("Model: {}".format(base_model))
    return base_model
