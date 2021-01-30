import torch
import gym
import envs
#from gym_minigrid.wrappers import VectorObsWrapper
from misc.linear_baseline import LinearFeatureBaseline, get_return
from misc.replay_memory import ReplayMemory
from misc.multiprocessing_env import SubprocVecEnv

iteration = 0


def make_env(args, env=None, task=None):
    def _make_env():
        env.reset_task(task=task)
        return env
    return _make_env


def collect_trajectory(task, base_model, args, logger):
    global iteration
    assert len(task) == 2, "Format should be (env, task)"

    # Initialize memory
    memory = ReplayMemory()

    # Set environment
    print("Collecting traj with task {}".format(task[1]))
    env = SubprocVecEnv([make_env(args, env=task[0], task=task[1]) for _ in range(args.batch[0])])
    print("Collecting succeeded")

    obs = env.reset()

    for timestep in range(args.ep_max_timestep):
        # Get action and logprob
        categorical = base_model(obs)
        action = categorical.sample()
        logprob = categorical.log_prob(action)
        if args.is_continuous_action:
            logprob = torch.sum(logprob, dim=1)

        # Take step in the environment
        if args.is_continuous_action:
            action = action.cpu().numpy().astype(float)
        else:
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

    env.close()

    iteration += 1

    return memory


def get_inner_loss(base_model, task, args, logger):
    memory = collect_trajectory(task, base_model, args, logger)
    obs, action, logprob, reward, mask = memory.sample()
    logprob = torch.stack(logprob, dim=1)

    # Get baseline
    linear_baseline = LinearFeatureBaseline(obs)
    value = linear_baseline(obs, reward, mask)

    # Get REINFORCE loss with baseline
    logprob = logprob * mask
    return_ = get_return(reward, mask)
    loss = torch.mean(torch.sum(logprob * (return_ - value), dim=1))

    return -loss, memory
