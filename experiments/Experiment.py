import sys
import time

import gym
import numpy as np
from stable_baselines3 import DQN
import yaml

from stable_baselines3.common.logger import make_output_format
from DQN.AdaptiveDQN import AdaptiveDQN
from environments.EnvWrapper import EnvWrapper
from environments.MazeMilestoneGenerator import MountainCarMilestoneGenerator, MazeMilestoneGenerator
from environments.Milestone import PassingMilestone, ExactMilestone
from environments.gym_maze import *
from environments.gym_maze.envs import MazeEnv
from DQN.uncertainty import CountUncertainty
import argparse


class Experiment:
    def __init__(self, num_milestones: int = 10, file:str ="mountaincar.yaml"):
        self.num_milestones = num_milestones
        self.config = self._read_yaml(file) if file != "parse" else self.parse_args()
        print("Starting Experiment.. config ", file)

    def _read_yaml(self, file: str):
        with open(file, "r") as f:
            return yaml.safe_load(f)

    def get_env(self):
        return gym.make(self.config['env'])

    def get_env_wrapper(self, env):
        raise NotImplementedError("You should implement the wrapper in a subclass!")

    def _train(self, model):
        model.learn(total_timesteps=self.config["trainsteps"])

    def _demo(self, env, model):
        # We have to force enable render for maze
        # TODO: there must be a cleaner way !
        try:
            env.env.env.do_enable_render()
        except AttributeError:
            pass

        obs = env.reset()
        for i in range(self.config["demosteps"]):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
            time.sleep(0.5)

    def main(self):
        """Example for running the AdaptiveDQN"""

        # Set seed for reproducibility
        seed = self.config['seed']
        # Start learning from the 0th timestep
        learning_starts = 0

        env = self.get_env_wrapper(self.get_env())
        uncertainty = CountUncertainty(env, **self.config[
            'uncertainty_kwargs']) if 'uncertainty_kwargs' in self.config else None

        model = AdaptiveDQN(env, self.config['policy'], env, eps_method=self.config['method'], plot=self.config['plot'],
                            decay_func=lambda x: np.sqrt(np.sqrt(x)), verbose=1, learning_starts=learning_starts,
                            seed=seed,
                            policy_kwargs=self.config['policy_kwargs'], uncertainty=uncertainty)

        self._train(model)
        self._demo(env, model)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help="Config file to use. Other settings will override the config "
                                                   "that is read in from the file", metavar="FILE",
                            default="mountaincar.yaml")
        args = parser.parse_args()

        config = yaml.safe_load(open(args.config))
        return config
