import gym
from gym_minigrid.wrappers import VectorObsWrapper


def make_env(args, env_name=None, task=None):
    # Set dummy task
    if env_name is None:
        env_name = "MiniGrid-Unlock-Easy-v0"
    if task is None:
        task = (4, 4)

    def _make_env():
        env = gym.make(env_name)
        env.max_steps = args.ep_max_timestep  # TODO Set horizon from config file
        env.reset_task(task=task)  # TODO Eliminate same task
        return VectorObsWrapper(env)        
    return _make_env
