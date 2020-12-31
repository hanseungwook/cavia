import gym
import numpy as np
from gym.envs.registration import register


# 2D Navigation
register(
    '2DNavigation-v0',
    entry_point='gym_env.navigation:Navigation2DEnv',
    max_episode_steps=100
)


# 2D Navigation Rotation
register(
    '2DNavigationRot-v0',
    entry_point='gym_env.navigation_rot:NavigationRot2DEnv',
    max_episode_steps=100
)


# 2D Navigation Acceleration
register(
    '2DNavigationAcc-v0',
    entry_point='gym_env.navigation_acc:NavigationAcc2DEnv',
    max_episode_steps=100
)


def make_env(args, env=None, task=None):
    # Set dummy task
    if env is None:
        env = gym.make("2DNavigation-v0")

    if task is None:
        task = {"goal": np.array([0, 0])}

    def _make_env():
        env.max_steps = args.ep_max_timestep
        env.reset_task(task=task)
        return env
    return _make_env
