from enum import Enum
from typing import Dict, Tuple

from scipy import interpolate

from DQN.uncertainty import CountUncertainty
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.logger import Figure

from environments.EnvWrapper import EnvWrapper
import numpy as np
import matplotlib.pyplot as plt
import os

from plotting.HeatMap import HeatMap


class ExplorationMethod(str, Enum):
    TRADITIONAL = "traditional"
    TRADITIONAL_WITH_MILESTONES = "traditional_milestones"
    ADAPTIVE_1 = "adaptive1"
    ADAPTIVE_2 = "adaptive2"
    DEEP_EXPLORATION = "deep_exploration"


class AdaptiveDQN(DQN):
    def __init__(self, env_wrapper: EnvWrapper, *args, results_folder, exploration_method: ExplorationMethod, plot,
                 eps_zero=1.0, decay_func=np.sqrt, uncertainty=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_wrapper = env_wrapper
        self.eps_zero = eps_zero
        self.decay_func = decay_func
        self.exploration_method = exploration_method
        self.plot = plot
        self.uncertainty = uncertainty
        self.path_to_results = results_folder

        if self.plot["enabled"]:
            self.exploration_array = []
            self.milestone_array = []
            self.reward_array = []
            self.episode_array = []
            self.fig, self.axis = plt.subplots(2, 3, figsize=(12, 10))
            self.heat_map = HeatMap(env_wrapper, uncertainty, axis=self.axis[0, 2])
            plt.ion()
            plt.show()


    def add_plot(self, axis, y, title="", per_episode=False, ylabel="", smooth=False):
        x = np.arange(1, len(y) + 1)

        if not smooth:
            axis.plot(x, y, 'g')
        else:
            # Smooth with moving average
            y_smooth = np.convolve(y, np.ones(self.plot["smooth"]["n"]) / self.plot["smooth"]["n"], 'valid')
            x_smooth = np.arange(1, len(y_smooth) + 1)
            axis.plot(x_smooth, y_smooth, 'g')

        if title is not None and len(title) > 0:
            axis.set_title(title)

        if per_episode:
            axis.set_xlabel("Episode")
        else:
            axis.set_xlabel("Timestep")

        if ylabel is not None and len(ylabel) > 0:
            axis.set_ylabel(ylabel)

    def plot_results(self):
        milestone_array = [ep["num_milestones_reached"] for ep in self.env_wrapper._episode_log]
        episode_reward_array = [ep["episode_rewards"] for ep in self.env_wrapper._episode_log]
        total_reward_array = [ep["total_rewards"] for ep in self.env_wrapper._episode_log]

        self.add_plot(self.axis[0, 0], y=self.exploration_array, title="Exploration rates",
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("exploration_rate")))
        self.add_plot(self.axis[0, 1], y=milestone_array, title="Reached milestones", per_episode=True,
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("milestones")))
        self.add_plot(self.axis[1, 0], y=episode_reward_array, title="Received rewards", per_episode=True,
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("episode_rewards")))
        self.add_plot(self.axis[1, 1], y=total_reward_array, title="Total rewards (inc. milestones)", per_episode=True,
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("total_rewards")))
        self.add_plot(self.axis[1, 2], y=self.episode_array, title="Elapsed episodes",
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("elapsed_episodes")))

        plt.tight_layout()
        plt.draw()
        plt.pause(0.3)

        file_path = os.path.join(self.path_to_results, "plot_results.pdf")
        if os.path.isfile(file_path):
            os.remove(file_path)
        plt.savefig(file_path)

    def _get_exploration_rate(self) -> float:
        """
        Gets the exploration rate, retrieving the current milestones
        and finding its respective epsilon
        :return: float (epsilon)
        """
        # Current number of milestones reached
        current_milestone = self.env_wrapper.get_number_of_milestones_reached()
        if self.exploration_method == ExplorationMethod.ADAPTIVE_1:
            return self.eps_zero / (self.decay_func(self.env_wrapper.counter[current_milestone]))
        elif self.exploration_method == ExplorationMethod.ADAPTIVE_2:
            # Method 2:
            #   Look at the next milestone in the list
            #   Add + 1 because we otherwise divide by zero at the start
            return self.eps_zero / (self.decay_func(self.env_wrapper.counter[current_milestone + 1]) + 1)
        else:
            # If exploration method is not an adaptive method, we just use the regular linear decay
            return self.exploration_schedule(self._current_progress_remaining)

    def _on_step(self):
        """Overwrite _on_step method from DQN class"""
        if self.env_wrapper.uses_milestones:
            self._n_calls += 1
            if self._n_calls % self.target_update_interval == 0:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

            self.exploration_rate = self._get_exploration_rate()
            self.logger.record("rollout/exploration_rate", self.exploration_rate)

        else:
            super()._on_step()

        if self.plot["enabled"] and self._n_calls % self.plot["update_interval"] == 0:
            self.plot_results()

            if "MountainCar" in self.env_wrapper.spec.id:
                self.heat_map.generate1D()
            if "maze" in self.env_wrapper.spec.id:
                self.heat_map.generate2D()
            if self.plot["reset_heat_map_every_update"]:
                self.heat_map.reset_count()

        if self.plot["enabled"]:
            self.exploration_array = np.append(self.exploration_array, self.exploration_rate)
            self.milestone_array = np.append(self.milestone_array, self.env_wrapper.get_number_of_milestones_reached())
            self.reward_array = np.append(self.reward_array, self.env_wrapper.reward)
            self.episode_array = np.append(self.episode_array, self._episode_num)

    #     if self.is_new_episode():
    #         print(f"new episode: {self._episode_num}")
    #         print(self.ep_info_buffer)
    #
    # def is_new_episode(self):
    #     return len(self.episode_array) > 2 and self.episode_array[-1] - self.episode_array[-2] >= 1

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        if self.uncertainty is not None:
            self.uncertainty.observe(new_obs)
        return super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
