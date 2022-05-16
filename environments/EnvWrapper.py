from typing import List

import gym
import numpy as np
from numpy.typing import NDArray

from environments.Milestone import Milestone


class EnvWrapper:
    """Wrapper class for a Gym environment that enables the milestone system."""

    def __init__(self, env: gym.Env, milestones: List[Milestone] = None, reward=0):
        self.env = env
        self.milestones = milestones if milestones else []
        self.n_milestones = len(milestones)
        self.reward = reward

        if self.n_milestones > 0:
            self.milestones_reached = np.zeros(self.n_milestones, dtype=np.bool)

            # Add a boolean variable for each milestone
            # We extend the original Box space of the gym environment
            # TODO: This might not work for every environment?
            new_low = np.append(self.env.observation_space.low, np.zeros(self.n_milestones, dtype=np.float32))
            new_high = np.append(self.env.observation_space.high, np.ones(self.n_milestones, dtype=np.float32))
            self.observation_space = gym.spaces.Box(low=new_low, high=new_high)
        else:
            self.observation_space = env.observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action=action)

        if self.n_milestones > 0:
            # Update the milestones if we are working with the milestone system
            extra_reward = self.update_milestones(obs)
            reward += extra_reward
            self.reward = reward
            obs = np.append(obs, self.milestones_reached.astype(np.float32))

        return obs, reward, done, info

    def update_milestones(self, state: NDArray) -> float:
        """
        Check for every milestone if it is reached.
        Return sum of rewards that are achieved.
        """
        # A milestone is reached if it is reached with the new state or it already was reached
        are_milestones_reached = np.array([milestone.is_achieved(state) or self.milestones_reached[i]
                                           for i, milestone in enumerate(self.milestones)])
        # Get the indices of all newly reached milestones
        new_milestones_reached = np.where(are_milestones_reached > self.milestones_reached)[0]

        # Sum the rewards of those milestones
        reward = sum(self.milestones[i].reward for i in new_milestones_reached)

        # Update the reached milestones and return the attained reward
        self.milestones_reached = are_milestones_reached
        return reward

    def reset(self):
        obs = self.env.reset()
        if self.n_milestones > 0:
            self.milestones_reached = np.zeros(self.n_milestones, dtype=np.bool)
            obs = np.append(obs, self.milestones_reached.astype(np.float32))
        return obs

    # Propagate all other functions to the environment
    def render(self, mode="human"):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def spec(self):
        return self.env.spec

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __str__(self):
        return self.env.__str__()

    def __enter__(self):
        return self.env.__enter__()

    def __exit__(self, *args):
        return self.env.__exit__()
