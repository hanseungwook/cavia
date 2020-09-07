import torch
import gym
import numpy as np
from replay_memory import ReplayMemory
from gym_minigrid.wrappers import VectorObsWrapper
from linear_baseline import LinearFeatureBaseline, get_return
from multiprocessing_env import SubprocVecEnv
from torch.distributions import Categorical


def vector_to_parameters(vector, parameters):
    from torch.nn.utils.convert_parameters import _check_param_device

    param_device = None

    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)

        num_param = param.numel()
        param.data.copy_(vector[pointer:pointer + num_param]
                         .view_as(param).data)

        pointer += num_param


def make_env(args, env_name=None, task=None):
    # Set dummy task
    if env_name is None:
        env_name = "MiniGrid-Empty-5x5-v0"
    if task is None:
        task = (3, 3)

    def _make_env():
        env = gym.make(env_name)
        env.max_steps = args.ep_max_timestep
        env.reset_task(task=task)
        return VectorObsWrapper(env)        
    return _make_env


def collect_trajectory(task, base_model, args):
    # Initialize memory
    memory = ReplayMemory()

    # Set environment
    # TODO Set from task
    # TODO Avoid hard-coding
    env = SubprocVecEnv([make_env(args) for _ in range(20)])

    obs = env.reset()
    score = 0.

    for timestep in range(args.ep_max_timestep):
        # Get action and logprob
        categorical = base_model(obs)
        action = categorical.sample()
        logprob = categorical.log_prob(action)

        # Take step in the environment
        action = action.cpu().numpy().astype(int)
        next_obs, reward, done, _ = env.step(action)

        # Add to memory
        memory.add(
            obs=obs, 
            action=torch.from_numpy(action), 
            logprob=logprob, 
            reward=reward,
            done=done)

        # For logging
        score += np.mean(reward)

        # For next timestep
        obs = next_obs

    print("score:", score)
    env.close()

    return memory


def get_inner_loss(base_model, task, args, memory=None):
    if memory is None:
        memory = collect_trajectory(task, base_model, args)
        obs, action, logprob, reward, mask = memory.sample()
        logprob = torch.stack(logprob, dim=1)
    else:
        obs, action, logprob, reward, mask = memory.sample()
        categorical = base_model(torch.from_numpy(np.stack(obs, axis=1)).float())
        action = torch.from_numpy(np.stack(action, axis=1)).float()
        logprob = categorical.log_prob(action)  # Replace logprob

    # Get baseline
    linear_baseline = LinearFeatureBaseline(obs)
    value = linear_baseline(obs, reward, mask)

    # Get REINFORCE loss with baseline
    logprob = logprob * mask
    return_ = get_return(reward, mask)
    loss = torch.mean(torch.sum(logprob * (return_ - value), dim=1))

    return -loss, memory


def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    else:
        raise NotImplementedError()

    return distribution
