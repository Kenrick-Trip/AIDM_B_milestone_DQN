from typing import List

import gym
import numpy as np
from numpy.typing import NDArray

from environments.Milestone import Milestone


class EnvWrapper:
    """Wrapper class for a Gym environment that enables the milestone system."""
    def __init__(self, env: gym.Env, milestones: List[Milestone] = None):
        self.env = env
        self.milestones = milestones if milestones else []
        self.n_milestones = len(milestones)
        self.reward = 0
        self.cached_milestones_reached = None
        self.counter = np.zeros(self.n_milestones, dtype=int)

        self._episode_rewards = np.array([])
        self._total_rewards = np.array([])
        self._episode_log = []

        if self.uses_milestones:
            self.milestones_reached = np.zeros(self.n_milestones, dtype=np.bool)

            # Add a boolean variable for each milestone
            # We extend the original Box space of the gym environment
            # TODO: This might not work for every environment?
            dtype = env.observation_space.dtype
            new_low = np.append(self.env.observation_space.low, np.zeros(self.n_milestones, dtype=dtype))
            new_high = np.append(self.env.observation_space.high, np.ones(self.n_milestones, dtype=dtype))
            self.observation_space = gym.spaces.Box(low=new_low, high=new_high)
        else:
            self.observation_space = env.observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action=action)
        self._episode_rewards = np.append(self._episode_rewards, reward)
        if self.uses_milestones:
            # Update the milestones if we are working with the milestone system
            extra_reward = self.update_milestones(obs)
            reward += extra_reward
            self.reward = reward

            # self.update_counter(self.get_curr_milestones())
            obs = np.append(obs, self.milestones_reached.astype(np.float32))
        self._total_rewards = np.append(self._total_rewards, reward)
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

    def get_curr_milestones(self) -> int:
        """
        Returns the current milestone i.e. index of last boolean true in reached milestones
        :return: integer (index)
        """
        if not self.uses_milestones:
            return 0
        indexes = np.where(self.milestones_reached)[0]
        return indexes[-1] if len(indexes) > 0 else 0

    def update_counter(self, milestones_reached):
        """
        Updates the counter, whenever the milestones_reached array changed, update the counter
        :param milestones_reached:
        """
        if self.cached_milestones_reached is not None and not np.array_equal(self.cached_milestones_reached,
                                                                             milestones_reached):
            self.counter[self.get_curr_milestones()] += 1

        self.cached_milestones_reached = milestones_reached

    def reset_counter(self):
        self.counter = np.zeros(self.n_milestones, dtype=int)

    def log_latest(self):
        if len(self._episode_rewards) == 0:
            return

        self._episode_log.append({
            "num_milestones_reached": int(self.milestones_reached.sum()) if self.uses_milestones else 0,
            "episode_rewards": self._episode_rewards.sum(),
            "total_rewards": self._total_rewards.sum()
        })

    def reset(self):
        self.log_latest()
        obs = self.env.reset()
        self._episode_rewards = np.array([])
        self._total_rewards = np.array([])
        if self.uses_milestones:
            self.counter[0] += 1
            self.milestones_reached = np.zeros(self.n_milestones, dtype=np.bool)
            obs = np.append(obs, self.milestones_reached.astype(np.float32))
        return obs

    def total_reset(self):
        obs = self.reset()
        self.cached_milestones_reached = None
        self.reset_counter()
        self._episode_log = []
        return obs

    # Propagate all other functions to the environment
    def render(self, mode="human"):
        if hasattr(self.env, "render_milestones"):
            self.env.render_milestones(self.milestones)
        if hasattr(self.env, "render_shortest_path"):
            self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed=seed)
        
    @property
    def uses_milestones(self):
        return self.n_milestones > 0

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
