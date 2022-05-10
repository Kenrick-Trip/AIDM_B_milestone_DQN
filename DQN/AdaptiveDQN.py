from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
from environments.EnvWrapper import EnvWrapper
import numpy as np


class AdaptiveDQN(DQN):
    def __init__(self, env_wrapper: EnvWrapper, *args, eps_zero=1.0, decay_func=np.sqrt, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_wrapper = env_wrapper
        self.counter = np.zeros(self.env_wrapper.n_milestones, dtype=int)
        self.eps_zero = eps_zero
        self.decay_func = decay_func

        self.cached_milestones_reached = None

    def update_counter(self, milestones_reached):
        """
        Updates the counter, whenever the milestones_reached array changed, update the counter
        :param milestones_reached:
        """
        if self.cached_milestones_reached is not None and not np.array_equal(self.cached_milestones_reached,
                                                                             milestones_reached):
            self.counter[self.get_curr_milestones()] += 1

        self.cached_milestones_reached = milestones_reached

    def get_eps(self):
        """
        Returns an array containing the epsilon for each milestone
        :return:
        """
        return self.eps_zero / (self.decay_func(self.counter) + 1)

    def get_curr_milestones(self) -> int:
        """
        Returns the current milestone i.e. index of last boolean true in reached miletones
        :return: integer (index)
        """
        indexes = np.where(self.env_wrapper.milestones_reached)[0]
        return indexes[-1] if len(indexes) > 0 else 0

    def _get_exploration_rate(self) -> float:
        """
        Gets the exploration rate, retrieving the current milestones
        and finding its respective epsilon
        :return: float (epsilon)
        """
        eps = self.get_eps()
        curr_milestone = self.get_curr_milestones()
        self.update_counter(curr_milestone)
        return eps[curr_milestone]

    def _on_step(self):
        """Overwrite _on_step method from DQN class"""
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self._get_exploration_rate()
        self.logger.record("rollout/exploration_rate", self.exploration_rate)
