# TODO Multiple tasks
import gym
import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_baseline import LinearFeatureBaseline, get_return
from torch.distributions.categorical import Categorical
from torch.optim import Adam, SGD
from gym_minigrid.wrappers import VectorObsWrapper
from replay_memory import ReplayMemory


def make_env(env_name):
    def _make_env():
        env = gym.make(env_name)
        env.max_steps = min(env.max_steps, 20)  # TODO Set horizon from config file
        return VectorObsWrapper(env)        
    return _make_env


max_iter_list = [3, 3, 20]  # Level 0, level 1, level 2
n_ctx = [2, 3]
n_input = 6 + sum(n_ctx)
lr = 0.01
layers = nn.Linear(n_input, 32), nn.Linear(32, 32), nn.Linear(32, 7)
data = torch.randn(10, 1), torch.randn(10, 1)
env = make_env("MiniGrid-Unlock-Easy-v0")()
env.unwrapped.reset_task(task=(4, 4))
baseline = LinearFeatureBaseline(6)


def collect_trajectory(base_model):
    # Initialize memory
    memory = ReplayMemory()

    obs = env.reset()

    for timestep in range(20):
        # Get action and logprob
        probs = base_model(obs)
        categorical = Categorical(probs=probs)
        action = categorical.sample()
        logprob = categorical.log_prob(action)

        # Take step in the environment
        action = action.cpu().numpy().astype(int)
        next_obs, reward, done, _ = env.step(action)

        # Add to memory
        memory.add(
            obs=obs, 
            logprob=logprob, 
            reward=reward,
            done=done)

        # For next timestep
        obs = next_obs

        if done:
            break

    return memory


def get_inner_loss(base_model, task):
    memory = collect_trajectory(base_model)  # TODO Collect multiple trajectories
    obs, logprob, reward, mask = memory.sample()

    # Get baseline
    value = baseline(obs, reward, mask)

    # Get REINFORCE loss with baseline
    logprob = torch.stack(logprob, dim=1)
    return_ = get_return(reward, mask)
    loss = torch.mean(torch.sum(logprob * (return_ - value), dim=1))

    return -loss


def make_ctx(n):
    return torch.zeros(1, n, requires_grad=True)


class BaseModel(nn.Module):
    def __init__(self, n_ctx, *layers):
        super().__init__()

        self.layers = nn.ModuleList(list(layers))        
        self.parameters_all = [make_ctx(n) for n in n_ctx] + [self.layers.parameters]
        self.nonlin = nn.ReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().unsqueeze(0)

        ctx = torch.cat(self.parameters_all[:-1], dim=1)
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.nonlin(x) if i < len(self.layers) - 1 else x

        return F.softmax(x)


class Hierarchical_Model(nn.Module):
    def __init__(self, submodel): 
        super().__init__()

        self.submodel = submodel 
        self.level_max = len(submodel.parameters_all)

    def forward(self, data, level=None, optimizer=None, reset=True): 
        if level is None:
            level = self.level_max

        if level == 0:
            return get_inner_loss(self.submodel, data)
        else:
            optimize(
                model=self, 
                data=data, 
                level=level - 1, 
                max_iter=max_iter_list[level - 1], 
                optimizer=optimizer, 
                reset=reset)
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
        print("loss:", loss)

        if reset:
            param_all[level], = optim.step(loss, params=[param_all[level]])
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()  


basemodel = BaseModel(n_ctx, *layers)
model = Hierarchical_Model(basemodel) 
model(data, optimizer=Adam, reset=False)
