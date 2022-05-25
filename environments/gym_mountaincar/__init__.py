from gym.envs.registration import register

register(
    id='MountainCarMilestones-v0',
    entry_point='environments.gym_mountaincar.envs:MountainCarMilestonesEnv',
    max_episode_steps=1000,
)