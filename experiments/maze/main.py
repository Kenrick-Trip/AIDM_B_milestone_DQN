from experiments.Experiment import Experiment
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


class MountainCarExperiment(Experiment):
    def get_env_wrapper(self, env):
        milestone_generator = MazeMilestoneGenerator(env)
        milestones = milestone_generator.get_milestones(self.num_milestones)
        return EnvWrapper(env, milestones)


if __name__ == '__main__':
    MountainCarExperiment(file="maze.yaml").main()
