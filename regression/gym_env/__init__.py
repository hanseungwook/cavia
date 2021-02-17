import gym
from gym.envs.registration import register

register(
    'HalfCheetahVel-v2',
    entry_point="gym_env.mujoco.half_cheetah_vel:HalfCheetahVelEnv",
)

register(
    'HalfCheetahDir-v2',
    entry_point="gym_env.mujoco.half_cheetah_dir:HalfCheetahDirEnv",
)


def make_env(args, env=None, task=None):
    if env is None:
        env = gym.make(args.task[0])

    def _make_env():
        env.max_steps = args.ep_max_timestep
        env.reset_task(task=task)
        return env
    return _make_env
