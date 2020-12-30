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
