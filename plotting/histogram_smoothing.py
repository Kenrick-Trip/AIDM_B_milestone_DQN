import itertools
import os
from typing import List, Any

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

COLORS = ["orange", "blue", "red", "green", "purple"]

class HistogramSmoothing:
    def __init__(self, result_dirs: List[str], parameter_name: str, parameter_values: List[Any], save_file=None,
                 n_bins=100):
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.save_file = save_file
        self.n_bins = n_bins

        self.bin_size = None
        self.bins = None
        self.binned_dfs = {}
        self.episode_statistics_df = None

        self.result_dfs, self.configs = self.read_results(result_dirs)
        self.max_reward = self.configs[0].get("max_reward")

    def read_results(self, result_dirs):
        dfs = {parameter_value: [] for parameter_value in self.parameter_values}
        configs = []
        for result_dir in result_dirs:
            config = yaml.safe_load(open(os.path.join(result_dir, "config.yaml")))
            configs.append(config)
            assert config[self.parameter_name] in self.parameter_values, \
                f"Unkown parameter value {config[self.parameter_name]} for parameter {self.parameter_name}"
            df = pd.read_csv(os.path.join(result_dir, "per_episode.csv"), index_col=0)
            dfs[config[self.parameter_name]].append(df)

        return dfs, configs

    def log(self, i, s):
        """Print a message s for a certain index i"""
        print(f"[{self.parameter_values[i]}] - {s}")

    def plot(self):
        for i, dfs in enumerate(self.result_dfs.values()):
            self.log(i, f"Number of runs: {len(dfs)}")
        self.make_bins()
        self.add_cum_episode_lengths()
        self.fill_bins()
        self.make_episode_statistics_per_bin()
        self.make_plot()

    def make_bins(self):
        n_timesteps = self.configs[0]["trainsteps"]
        assert all(n_timesteps == config["trainsteps"] for config in self.configs[1:]), \
            "All results must have same number of timesteps"
        self.bin_size = n_timesteps / self.n_bins
        assert int(self.bin_size) == self.bin_size, "Number of timesteps must be divisible by n_bins"
        self.bin_size = int(self.bin_size)
        self.bins = np.cumsum(np.full(self.n_bins, self.bin_size))

    def get_bin(self, timestep):
        """Get the bin of a timestep"""
        return timestep // self.bin_size

    def iterate_all_result_dfs(self):
        """Iterate over all result dataframe in the dictionary"""
        for name, dfs in self.result_dfs.items():
            for df in dfs:
                yield df

    def add_cum_episode_lengths(self):
        for result_df in self.iterate_all_result_dfs():
            result_df["cum_episode_length"] = result_df["episode_length"].cumsum()

    def fill_bins(self):
        # Filled bins contains for each bin a list
        # This list contains a list of episode numbers for each result
        for label, dfs in self.result_dfs.items():
            all_df_labels = [f"{label}_{i}" for i in range(len(dfs))]
            values = {}
            for df, df_label in zip(dfs, all_df_labels):
                mean_per_bin = np.full(self.n_bins, np.nan)
                cum_episode_length = df["cum_episode_length"].values
                for j, bin_end in enumerate(self.bins):
                    idx = np.bitwise_and((bin_end - self.bin_size) <= cum_episode_length, cum_episode_length < bin_end)
                    mean_value = df.loc[idx, "reward"].mean()
                    mean_per_bin[j] = mean_value
                values[df_label] = mean_per_bin
            self.binned_dfs[label] = pd.DataFrame(values)

    def make_episode_statistics_per_bin(self):
        statistics = ["mean", "std", "min", "max"]
        values = {}
        for label, df in self.binned_dfs.items():
            for statistic in statistics:
                statistic_label = f"{label}_{statistic}"
                statistics_per_bin = getattr(df, statistic)(axis=1).values
                values[statistic_label] = statistics_per_bin
        self.episode_statistics_df = pd.DataFrame(values)

    def make_plot(self):
        fig, ax = plt.subplots()
        for label, color in zip(self.parameter_values, COLORS[:len(self.parameter_values)]):
            y_min = np.maximum(self.episode_statistics_df[f"{label}_min"],
                               self.episode_statistics_df[f"{label}_mean"] - self.episode_statistics_df[f"{label}_std"])
            if self.max_reward is None:
                y_max = np.minimum(self.episode_statistics_df[f"{label}_max"],
                                   self.episode_statistics_df[f"{label}_mean"] + self.episode_statistics_df[
                                       f"{label}_std"])
            else:
                y_max = np.minimum(self.max_reward,
                                   self.episode_statistics_df[f"{label}_mean"] + self.episode_statistics_df[
                                       f"{label}_std"])
            ax.plot(self.bins, self.episode_statistics_df[f"{label}_mean"], label=label, color=color)
            ax.plot(self.bins, y_max, color=color, linestyle="dotted", alpha=0.2)
            ax.plot(self.bins, y_min, color=color, linestyle="dotted", alpha=0.2)
            ax.fill_between(self.bins, y_min, y_max, alpha=0.15, color=color)
        if self.max_reward is not None:
            ax.plot(self.bins, np.full(self.n_bins, self.max_reward), label=f"max_reward ({self.max_reward})")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Timestep")
        plt.legend()
        plt.savefig(self.save_file)
        plt.show()


def main():
    # base_dir = "../experiments/maze/results_benchmark1"
    base_dir = "../experiments/maze/results_maze"
    result_dirs = [
        "2022-06-05-t-150946",
        "2022-06-05-t-151313",
        "2022-06-05-t-162814",
        "2022-06-05-t-164722",
        "2022-06-05-t-171919",
        "2022-06-05-t-181534",
        "2022-06-05-t-182623",
        "2022-06-05-t-182759",
        "2022-06-05-t-182930",
        "2022-06-05-t-183229",
        "2022-06-05-t-183415",
        "2022-06-05-t-183600",
        # os.listdir(base_dir)
    ]
    result_dirs = [os.path.join(base_dir, result_dir) for result_dir in result_dirs]
    save_dir = os.path.join(base_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "plot.png")
    hs = HistogramSmoothing(result_dirs, "exploration_method",
                            ["adaptive4", "traditional_milestones", "traditional", "adaptive3"], save_file)
    hs.plot()


if __name__ == "__main__":
    main()
