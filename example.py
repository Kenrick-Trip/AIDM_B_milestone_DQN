import sys
import time

import gym
import numpy as np
from stable_baselines3 import DQN
import yaml

from stable_baselines3.common.logger import make_output_format
from DQN.AdaptiveDQN import AdaptiveDQN
from environments.EnvWrapper import EnvWrapper
from environments.MazeMilestoneGenerator import MazeMilestonesGenerator
from environments.Milestone import PassingMilestone, ExactMilestone
from environments.gym_maze import *
from environments.gym_maze.envs import MazeEnv


def read_yaml():
    with open("maze.yaml", "r") as f:
        return yaml.safe_load(f)


def set_milestones(config):
    # Create some example milestones
    if config['ENV'] == "MountainCar-v0":
        milestones = [
            # PassingMilestone(10, np.array([np.nan, 0.01])),  # Milestone for reaching a velocity of 0.01
            # PassingMilestone(100, np.array([-0.4, np.nan]))  # Milestone for reaching a position of -0.4
            PassingMilestone(1, np.array([-0.4, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(2, np.array([-0.3, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(4, np.array([-0.2, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(5, np.array([-0.1, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(6, np.array([0.0, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(8, np.array([0.1, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(9, np.array([0.2, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(10, np.array([0.3, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(11, np.array([0.4, np.nan])),  # Milestone for reaching a position of -0.4
            PassingMilestone(12, np.array([0.5, np.nan]))  # Milestone for reaching a position of -0.4
            # PassingMilestone(100, np.array([-0.5, np.nan]))  # Milestone for reaching a position of -0.4
        ]
        return milestones
    else:
        print('ERROR: no milestones defined for this environment')


def main(config, milestones):
    """Example for running the AdaptiveDQN"""

    # Set seed for reproducibility
    seed = config['SEED']
    # Start learning from the 0th timestep
    learning_starts = 0

    env = gym.make(config['ENV'])
    if "maze" in config["ENV"].lower():
        num_milestones = 4
        milestone_generator = MazeMilestonesGenerator(env)
        milestones = milestone_generator.calculate_milestones(num_milestones)
    env = EnvWrapper(env, milestones)
    model = AdaptiveDQN(env, config['POLICY'], env, eps_method=config['EPS_METHOD'], plot=config['PLOT'],
                        decay_func=lambda x: np.sqrt(np.sqrt(x)), verbose=1, learning_starts=learning_starts, seed=seed)
    model.learn(total_timesteps=config['TIMESTEPS'])

    # We have to force enable render for maze
    try:
        env.env.env.do_enable_render()
    except AttributeError:
        pass

    obs = env.reset()
    for i in range(config['DEMO_STEPS']):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        time.sleep(0.5)


    env.close()


if __name__ == '__main__':
    config_data = read_yaml()
    milestones_selected = set_milestones(config_data)
    main(config_data, milestones_selected)
