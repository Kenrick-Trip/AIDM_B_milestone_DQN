import argparse
import os
import time

import gym
import numpy as np
import yaml

from DQN.AdaptiveDQN import AdaptiveDQN
from environments.EnvWrapper import EnvWrapper
from environments.MazeMilestoneGenerator import MazeMilestoneGenerator, MountainCarMilestoneGenerator


class Runner:
    def __init__(self, config):
        self.config = config

        # Start learning from the 0th timestep
        learning_starts = 0

        self.env = gym.make(config.env)

        self.milestones = []
        self.generate_milestones()

        print(self.milestones)

        # Wrap environment
        self.env = EnvWrapper(self.env, self.milestones)

        self.model = AdaptiveDQN(self.env, config.policy, self.env, eps_method=config.method, plot=config.plot,
                            decay_func=lambda x: np.sqrt(np.sqrt(x)), verbose=1, learning_starts=learning_starts,
                            seed=config.seed)

    def generate_milestones(self):
        if "maze" in self.config.env.lower():
            num_milestones = 4
            milestone_generator = MazeMilestoneGenerator(self.env)
        else:
            num_milestones = 10
            milestone_generator = MountainCarMilestoneGenerator(self.env)
        self.milestones = milestone_generator.get_milestones(num_milestones)

    def train(self):
        self.model.learn(total_timesteps=self.config.trainsteps)

    def demo(self):
        # We have to force enable render for maze
        # TODO: there must be a cleaner way !
        try:
            self.env.env.env.do_enable_render()
        except AttributeError:
            pass

        obs = self.env.reset()
        for i in range(self.config.demosteps):
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
    parser.add_argument("-s", "--seed", help="Random seed to set", default=42)
    parser.add_argument("-t", "--trainsteps", help="Number of training steps", default=10000)
    parser.add_argument("-d", "--demosteps", help="Number of demo steps", default=1000)
    parser.add_argument("-p", "--policy", help="Stable baselines policy to use", default="MlpPolicy")
    parser.add_argument("-e", "--env", help="Gym environment", default="maze-random-5x5-v0")
    parser.add_argument("-m", "--method", choices={0, 1, 2}, help="Which exploration method to use, use 0 for naive "
                                                                  "epsilon greedy", default=1)
    parser.add_argument("-pl", "--plot", help="Create a plot", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = yaml.safe_load(open(args.config))
    else:
        print(f"Could not find {args.config}, using default values")
        config = {}

    for arg in vars(args):
        if arg == "config":
            continue
        else:
            config[arg] = getattr(args, arg)
    return args


def main(config):
    runner = Runner(config)
    runner.train()
    runner.demo()


if __name__ == "__main__":
    args = parse_args()
    main(args)
