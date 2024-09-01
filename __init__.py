import gymnasium

gymnasium.register(
    id='TrajectoryEnv',
    entry_point='trajectoryenv:TrajectoryEnv',
    autoreset=True,
    max_episode_steps=10000,
)
