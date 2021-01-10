import torch
import numpy as np
import torch.nn.functional as F
from gym_env import make_env
from misc.replay_memory import ReplayMemory
from misc.multiprocessing_env import SubprocVecEnv


def collect_trajectory(base_model, task, ctx, args):
    assert len(task) == 2, "Format should be (env, task)"

    # Initialize memory
    memory = ReplayMemory(args)

    # Set environment
    env = SubprocVecEnv([make_env(args, env=task[0], task=task[1]) for _ in range(args.batch[0])])

    # Collect trajectory
    obs = env.reset()
    for timestep in range(args.ep_max_timestep):
        # Get action and logprob
        distribution = base_model(obs, ctx)
        action = distribution.sample()
        logprob = distribution.log_prob(action)
        if args.is_continuous_action:
            logprob = torch.sum(logprob, dim=1)

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

        # For next timestep
        obs = next_obs

        if np.sum(done) >= args.batch[0]:
            break

    env.close()

    return memory


def get_inner_loss(base_model, task, ctx, args):
    memory = collect_trajectory(base_model, task, ctx, args)
    obs, action, logprob, reward, mask = memory.sample()
    logprob = torch.stack(logprob, dim=1)

    # Compute baseline
    value = base_model.baseline(obs, reward, mask, args)

    # Compute GAE
    # TODO Consider mask in padding zero
    value = F.pad(value * mask, (0, 1, 0, 0))  # Add an additional 0 at the end of episode
    deltas = torch.stack(reward, dim=1) + args.discount * value[:, 1:] - value[:, :-1]
    advantage = torch.zeros_like(deltas).float()
    gae = torch.zeros_like(deltas[:, 0]).float()
    for i in range(args.ep_max_timestep - 1, -1, -1):
        gae = gae * args.discount * args.tau + deltas[:, i]
        advantage[:, i] = gae

    # Get REINFORCE loss with GAE
    loss = -torch.mean(torch.sum(logprob * advantage.detach(), dim=1))

    return loss, memory


def get_return(reward, mask, args):
    if isinstance(reward, list):
        reward = torch.stack(reward, dim=1).float() * mask

    R, return_ = 0., []
    for timestep in reversed(range(reward.shape[-1])):
        R = reward[:, timestep] + args.discount * R
        return_.insert(0, R)
    return_ = torch.stack(return_, dim=1) * mask
    assert reward.shape == return_.shape

    return return_
