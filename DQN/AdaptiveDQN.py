from enum import Enum
from typing import Dict, Tuple

import pandas as pd
from scipy import interpolate

from DQN.uncertainty import CountUncertainty
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.logger import Figure
import torch as th
from torch.nn import functional as F

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
    ADAPTIVE_3 = "adaptive3"
    ADAPTIVE_4 = "adaptive4"
    DEEP_EXPLORATION = "deep_exploration"


class AdaptiveDQN(DQN):
    def __init__(self, env_wrapper: EnvWrapper, *args, results_folder, exploration_method: ExplorationMethod, config,
                 decay_func=np.sqrt, uncertainty=None, \
                                                                        intrinsic_reward=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.env_wrapper = env_wrapper
        self.eps_zero = config["eps_zero"]
        self.decay_func = decay_func
        self.exploration_method = exploration_method
        self.intrinsic_reward = intrinsic_reward
        self.plot = config["plot"]
        self.log = config["log"]
        self.uncertainty = uncertainty
        self.eps_min = config["eps_min"]
        self.denominator = config["denominator"]

        self.path_to_results = results_folder
        self.path_to_results_timestep = os.path.join(self.path_to_results, "per_timestep.csv")
        """Folder to save results per timestep"""
        self.path_to_results_episode = os.path.join(self.path_to_results, "per_episode.csv")
        """Folder to save results per episode"""

        # maze specific
        min_distance_path = self.path_to_results+"/tot_dist.tmp"
        if os.path.isfile(min_distance_path):
            f = open(min_distance_path, "r").read().splitlines()
            self.max_reward = 1 -0.1/float(f[0])*int(f[1])


        if self.log["enabled"] or self.plot["enabled"]:
            self.exploration_array = np.zeros(config["trainsteps"])
            self.milestone_array = np.zeros(config["trainsteps"])
            self.reward_array = np.zeros(config["trainsteps"])
            self.episode_array = np.zeros(config["trainsteps"])

            self.episode_milestone_array = []
            self.episode_reward_array = []
            self.episode_total_reward_array = []

        if self.plot["enabled"]:
            self.fig, self.axis = plt.subplots(2, 3, figsize=(12, 10))
            self.heat_map = HeatMap(env_wrapper, uncertainty, self.fig, axis=self.axis[0, 2])
            plt.ion()
            self.fig.show()

    def add_plot(self, axis, y, title="", per_episode=False, ylabel="", smooth=False,
                 plotMaxRew=False):
        # How to clear?
        # axis.clear()
        x = np.arange(1, len(y) + 1)

        if not smooth:
            axis.plot(x, y, 'g')

            if plotMaxRew and hasattr(self, 'max_reward'):
                axis.plot(x, [self.max_reward]*len(x), 'b')
        else:
            # Smooth with moving average
            y_smooth = np.convolve(y, np.ones(self.plot["smooth"]["n"]) / self.plot["smooth"]["n"], 'valid')
            x_smooth = np.arange(1, len(y_smooth) + 1)
            axis.plot(x_smooth, y_smooth, 'g')

            if plotMaxRew and hasattr(self, 'max_reward'):
                    axis.plot(x_smooth, [self.max_reward]*len(x_smooth), 'b')

        if title is not None and len(title) > 0:
            axis.set_title(title)

        if per_episode:
            axis.set_xlabel("Episode")
        else:
            axis.set_xlabel("Timestep")

        if ylabel is not None and len(ylabel) > 0:
            axis.set_ylabel(ylabel)

        plt.setp(axis.get_xticklabels(), horizontalalignment='right',
                 fontsize='x-small')

    def update_episode_arrays(self):
        self.episode_milestone_array = np.array([ep["num_milestones_reached"] for ep in self.env_wrapper._episode_log])
        self.episode_reward_array = np.array([ep["episode_rewards"] for ep in self.env_wrapper._episode_log])
        self.episode_total_reward_array = np.array([ep["total_rewards"] for ep in self.env_wrapper._episode_log])

    def plot_results(self):
        self.add_plot(self.axis[0, 0], y=self.exploration_array[:self._n_calls - 1], title="Exploration rates",
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("exploration_rate")))
        self.add_plot(self.axis[0, 1], y=self.episode_milestone_array, title="Reached milestones", per_episode=True,
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("milestones")))
        self.add_plot(self.axis[1, 0], y=self.episode_reward_array, title="Received rewards", per_episode=True,
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get(
                          "episode_rewards")), plotMaxRew=True)
        self.add_plot(self.axis[1, 1], y=self.episode_total_reward_array, title="Total rewards (inc. milestones)", per_episode=True,
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("total_rewards")))
        self.add_plot(self.axis[1, 2], y=self.episode_array[:self._n_calls - 1], title="Elapsed episodes",
                      smooth=self.plot["smooth"]["enabled"] and bool(self.plot["smooth"].get("elapsed_episodes")))

    def draw_plot(self):
        plt.tight_layout()
        plt.draw()
        plt.pause(0.3)

        file_path = os.path.join(self.path_to_results, "plot_results.pdf")
        if os.path.isfile(file_path):
            os.remove(file_path)
        # plt.savefig(file_path)
        self.fig.savefig(file_path)
        self.fig.show()

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
        elif self.exploration_method == ExplorationMethod.ADAPTIVE_3:
            # Method 3
            #   Look at the current milestone in the list, but use a linear decay function
            return max(self.eps_zero*(1 - self.env_wrapper.counter[
                current_milestone]/self.denominator),
                       self.eps_min)
        elif self.exploration_method == ExplorationMethod.ADAPTIVE_4:
            # Method 4
            #   Look at the current milestone in the list, but use a linear decay function
            return max(self.eps_zero*(1 - self.env_wrapper.counter[
                current_milestone + 1]/self.denominator),
                       self.eps_min)
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

            # Add data to the logger:
            n = len(self.env_wrapper._episode_log)
            self.logger.record("rollout/num_milestones_reached_per_episode",
                               0 if n == 0 else self.env_wrapper._episode_log[n - 1]["num_milestones_reached"])
            self.logger.record("rollout/episode_rewards",
                               0 if n == 0 else self.env_wrapper._episode_log[n - 1]["episode_rewards"])
            self.logger.record("rollout/total_rewards",
                               0 if n == 0 else self.env_wrapper._episode_log[n - 1]["total_rewards"])

        else:
            super()._on_step()

        if self.plot["enabled"] and self._n_calls % self.plot["update_interval"] == 0 \
            or self.log["enabled"] and self._n_calls % self.log["save_interval"] == 0:
            self.update_episode_arrays()

        if self.plot["enabled"] or self.log["enabled"]:
            self.exploration_array[self._n_calls - 1] = self.exploration_rate
            self.milestone_array[self._n_calls - 1] = self.env_wrapper.get_number_of_milestones_reached()
            self.reward_array[self._n_calls - 1] = self.env_wrapper.reward
            self.episode_array[self._n_calls - 1] = self._episode_num

        if self.plot["enabled"] and self._n_calls % self.plot["update_interval"] == 0:
            self.plot_results()

            if "MountainCar" in self.env_wrapper.spec.id or "MountainCarMilestones" in self.env_wrapper.spec.id:
                self.heat_map.generate1D()
            if "maze" in self.env_wrapper.spec.id:
                self.heat_map.generate2D()
            if self.plot["reset_heat_map_every_update"]:
                self.heat_map.reset_count()

            self.draw_plot()

        if self.log["enabled"] and self._n_calls % self.log["save_interval"] == 0:
            self.save_log_values()

    def save_log_values(self):
        # Save values that are updated every timestep
        timestep_df = pd.DataFrame({
            "exploration_rate": self.exploration_array[:self._n_calls - 1],
            "milestones_reached": self.milestone_array[:self._n_calls - 1],
            "reward": self.reward_array[:self._n_calls - 1],
            "episode_num": self.episode_array[:self._n_calls - 1]
        })
        timestep_df.to_csv(self.path_to_results_timestep)

        # Save values that are updated every episode
        episode_df = pd.DataFrame({
            "milestones_reached": self.episode_milestone_array,
            "reward": self.episode_reward_array,
            "total_reward": self.episode_total_reward_array,
        })
        episode_df.to_csv(self.path_to_results_episode)

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        if self.uncertainty is not None:
            self.uncertainty.observe(new_obs)
        return super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Add intrinsic reward based on the uncertainty counts
            rewards = replay_data.rewards
            if self.intrinsic_reward:
                rewards += self.uncertainty(replay_data.next_observations).unsqueeze(
                    dim=-1)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
