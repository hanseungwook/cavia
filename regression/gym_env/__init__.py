import gym
import numpy as np
from gym.envs.registration import register


# 1D Navigation
register(
    '1DNavigation-v0',
    entry_point='gym_env.navigation_1d:Navigation1DEnv',
    max_episode_steps=25
)


# 1D Navigation Rotation
register(
    '1DNavigationRot-v0',
    entry_point='gym_env.navigation_rot:Navigation1DRotEnv',
    max_episode_steps=25
)


# 2D Navigation
register(
    '2DNavigation-v0',
    entry_point='gym_env.navigation:Navigation2DEnv',
    max_episode_steps=50
)


# 2D Navigation Acceleration
register(
    '2DNavigationAcc-v0',
    entry_point='gym_env.navigation_acc:NavigationAcc2DEnv',
    max_episode_steps=50
)


def make_env(args, env=None, task=None):
    # Set dummy task
    if env is None:
        env = gym.make("1DNavigationRot-v0")

    if task is None:
        task = {"goal": np.array([0, 0])}

    def _make_env():
        env.max_steps = args.ep_max_timestep
        env.reset_task(task=task)
        return env
    return _make_env
