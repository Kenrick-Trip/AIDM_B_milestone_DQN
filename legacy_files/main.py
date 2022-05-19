import argparse
import os
import time

import gym
import numpy as np
import yaml

from DQN.AdaptiveDQN import AdaptiveDQN
from DQN.uncertainty import CountUncertainty
from environments.EnvWrapper import EnvWrapper
from environments.MilestoneGenerator import MazeMilestoneGenerator, MountainCarMilestoneGenerator


class Runner:
    def __init__(self, config):
        self.config = config

        # Start learning from the 0th timestep
        learning_starts = 0

        self.env = gym.make(config["env"])

        self.milestones = []
        self.generate_milestones()

        # Wrap environment
        self.env = EnvWrapper(self.env, self.milestones)
        self.uncertainty = CountUncertainty(self.env, **config['uncertainty_kwargs']) if 'uncertainty_kwargs' in config else None
        self.model = AdaptiveDQN(self.env, config["policy"], self.env, eps_method=config["method"], plot=config["plot"],
                                 decay_func=lambda x: np.sqrt(np.sqrt(x)), verbose=1, learning_starts=learning_starts,
                                 seed=config["seed"], uncertainty=self.uncertainty,
                                 plot_update_interval=config["plot_update_interval"],
                                 reset_heat_map_every_plot=config["reset_heat_map_every_plot"])

    def generate_milestones(self):
        if "maze" in self.config["env"].lower():
            milestone_generator = MazeMilestoneGenerator(self.env)
        else:
            milestone_generator = MountainCarMilestoneGenerator(self.env)
        self.milestones = milestone_generator.get_milestones(self.config["num_milestones"])
        print(self.milestones)

    def train(self):
        self.model.learn(total_timesteps=self.config["trainsteps"])

    def demo(self):
        # We have to force enable render for maze
        # TODO: there must be a cleaner way !
        try:
            self.env.env.env.do_enable_render()
        except AttributeError:
            pass

        obs = self.env.reset()
        for i in range(self.config["demosteps"]):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
            time.sleep(0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file to use. Other settings will override the config "
                                               "that is read in from the file", metavar="FILE", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    return config


def main(config):
    runner = Runner(config)
    runner.train()
    runner.demo()


if __name__ == "__main__":
    args = parse_args()
    main(args)
