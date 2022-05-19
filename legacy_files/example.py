import sys
import time

import gym
import numpy as np
from stable_baselines3 import DQN
import yaml

from stable_baselines3.common.logger import make_output_format
from DQN.AdaptiveDQN import AdaptiveDQN
from environments.EnvWrapper import EnvWrapper
from environments.MilestoneGenerator import MountainCarMilestoneGenerator, MazeMilestoneGenerator
from environments.Milestone import PassingMilestone, ExactMilestone
from environments.gym_maze import *
from environments.gym_maze.envs import MazeEnv
from DQN.uncertainty import CountUncertainty

def read_yaml():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main(config):
    """Example for running the AdaptiveDQN"""

    # Set seed for reproducibility
    seed = config['seed']
    # Start learning from the 0th timestep
    learning_starts = 0

    env = gym.make(config['env'])
    if "maze" in config["env"].lower():
        num_milestones = 4
        milestone_generator = MazeMilestoneGenerator(env)
    else:
        num_milestones = 10
        milestone_generator = MountainCarMilestoneGenerator(env)
    milestones = milestone_generator.get_milestones(num_milestones)
    env = EnvWrapper(env, milestones)
    uncertainty = CountUncertainty(env, **config['uncertainty_kwargs']) if 'uncertainty_kwargs' in config else None

    model = AdaptiveDQN(env, config['policy'], env, eps_method=config['method'], plot=config['plot'],
                        decay_func=lambda x: np.sqrt(np.sqrt(x)), verbose=1, learning_starts=learning_starts, seed=seed,
                        policy_kwargs=config['policy_kwargs'], uncertainty=uncertainty)
    model.learn(total_timesteps=config['trainsteps'])

    # We have to force enable render for maze
    try:
        env.env.env.do_enable_render()
    except AttributeError:
        pass

    obs = env.reset()
    for i in range(config['demosteps']):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        time.sleep(0.5)

    env.close()


if __name__ == '__main__':
    config_data = read_yaml()
    main(config_data)
