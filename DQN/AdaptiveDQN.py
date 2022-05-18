from DQN.uncertainty import CountUncertainty
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.logger import Figure

from environments.EnvWrapper import EnvWrapper
import numpy as np
import matplotlib.pyplot as plt
import os

from plotting.HeatMap import HeatMap


class AdaptiveDQN(DQN):
    def __init__(self, env_wrapper: EnvWrapper, *args, results_folder, eps_method, plot, eps_zero=1.0, decay_func=np.sqrt,
                 uncertainty=None, plot_update_interval=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_wrapper = env_wrapper
        self.eps_zero = eps_zero
        self.decay_func = decay_func
        self.method = eps_method
        self.plot = plot
        self.uncertainty = uncertainty
        self.plot_update_interval = plot_update_interval
        self.path_to_results = results_folder

        if self.plot == 1:
            self.exploration_array = []
            self.milestone_array = []
            self.reward_array = []
            self.episode_array = []
            self.fig, self.axis = plt.subplots(2, 3)
            self.heat_map = HeatMap(env_wrapper, uncertainty, axis=self.axis[0, 2])
            plt.ion()
            plt.show()

    def get_eps(self):
        """
        Returns an array containing the epsilon for each milestone
        :return:
        """
        return self.eps_zero / (self.decay_func(self.env_wrapper.counter) + 1)

    def plot_results(self):
        self.axis[0, 0].plot(self.exploration_array, 'g')
        self.axis[0, 0].set_title('Exploration rates')
        self.axis[0, 1].plot(self.milestone_array, 'g')
        self.axis[0, 1].set_title('Reached milestones')
        self.axis[1, 0].plot(self.reward_array, 'g')
        self.axis[1, 0].set_title('Received reward')
        self.axis[1, 1].plot(self.episode_array, 'g')
        self.axis[1, 1].set_title('Elapsed episodes')

        self.logger.record("trajectory/figure", Figure(self.fig, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.tight_layout()
        plt.draw()
        plt.pause(0.3)

        if self._n_calls > self.num_timesteps - self.plot_update_interval/2:
            file_path = "./{}/plot_results.pdf".format(self.path_to_results)
            if os.path.isfile(file_path):
                os.remove(file_path)
            plt.savefig(file_path)

    def _get_exploration_rate(self) -> float:
        """
        Gets the exploration rate, retrieving the current milestones
        and finding its respective epsilon
        :return: float (epsilon)
        """
        eps = self.get_eps()
        curr_milestone = self.env_wrapper.get_curr_milestones()

        if self.method == 1:
            self.env_wrapper.update_counter(curr_milestone)
            return eps[curr_milestone]
        elif self.method == 2:
            self.env_wrapper.update_counter(curr_milestone)
            return eps[curr_milestone + 1]
        else:
            raise ValueError("ERROR: epsilon selection method is not valid, must be 1 or 2")

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

        if self._n_calls % self.plot_update_interval == 0:
            if self.plot == 1:
                self.plot_results()
                self.heat_map.generate1D()
                self.heat_map.reset_count()

        if self.plot == 1:
            self.exploration_array = np.append(self.exploration_array, self.exploration_rate)
            self.milestone_array = np.append(self.milestone_array, self.env_wrapper.get_curr_milestones())
            self.reward_array = np.append(self.reward_array, self.env_wrapper.reward)
            self.episode_array = np.append(self.episode_array, self._episode_num)

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        if self.uncertainty is not None:
            self.uncertainty.observe(new_obs)
        return super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
