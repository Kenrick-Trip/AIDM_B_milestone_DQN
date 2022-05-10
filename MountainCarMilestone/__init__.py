# import logging
from gym.envs.registration import register

# logger = logging.getLogger(__name__)

register(
    id='MountainCarMilestones-v0',
    entry_point='MountainCarMilestone.environments:MountainCarMilestonesEnv',
    max_episode_steps=1000,
)