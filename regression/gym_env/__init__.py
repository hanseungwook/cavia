from gym.envs.registration import register


# 2D Navigation
# ----------------------------------------
register(
    '2DNavigation-v0',
    entry_point='gym_env.navigation:Navigation2DEnv',
    max_episode_steps=100
)
