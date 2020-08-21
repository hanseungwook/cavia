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
from multiprocessing_env import SubprocVecEnv
from tensorboardX import SummaryWriter

traj_batch_size = 15
ep_max_timestep = 20

tb_writer = SummaryWriter('./logs/tb_{0}'.format("rl_merge"))


def make_env(env_name):
    def _make_env():
        env = gym.make(env_name)
        env.max_steps = min(env.max_steps, ep_max_timestep)  # TODO Set horizon from config file
        env.reset_task(task=(4, 4))  # TODO Eliminate same task
        return VectorObsWrapper(env)        
    return _make_env


max_iter_list = [3, 3, 500]  # Level 0, level 1, level 2
n_ctx = [3, 3]
n_input = 6 + sum(n_ctx)  # TODO Remove hard-coding
lr = 0.005
layers = nn.Linear(n_input, 64), nn.Linear(64, 64), nn.Linear(64, 7)  # TODO Remove hard-coding
data = torch.randn(10, 1), torch.randn(10, 1)
env = SubprocVecEnv([make_env("MiniGrid-Unlock-Easy-v0") for _ in range(traj_batch_size)])
linear_baseline = LinearFeatureBaseline(6)  # TODO Remove hard-coding
counter = 0


def collect_trajectory(base_model):
    global counter

    # Initialize memory
    memory = ReplayMemory()

    obs = env.reset()
    score = 0.

    for timestep in range(ep_max_timestep):
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

        # For logging
        score += np.mean(reward)

        # For next timestep
        obs = next_obs

    print("score:", score)
    counter += 1
    tb_writer.add_scalar("score", score, counter)

    return memory


def get_inner_loss(base_model, task):
    memory = collect_trajectory(base_model)  # TODO Collect multiple trajectories
    obs, logprob, reward, mask = memory.sample()

    # Get baseline
    value = linear_baseline(obs, reward, mask)

    # Get REINFORCE loss with baseline
    logprob = torch.stack(logprob, dim=1) * mask
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
            x = torch.from_numpy(x).float()

        ctx = torch.cat(self.parameters_all[:-1], dim=1)
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.nonlin(x) if i < len(self.layers) - 1 else x

        return F.softmax(x, dim=1)


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

        if reset:
            param_all[level], = optim.step(loss, params=[param_all[level]])
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()  


basemodel = BaseModel(n_ctx, *layers)
model = Hierarchical_Model(basemodel) 
model(data, optimizer=Adam, reset=False)
