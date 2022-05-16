import sys
import time

import gym
import numpy as np
from stable_baselines3 import DQN

from DQN.AdaptiveDQN import AdaptiveDQN
from environments.EnvWrapper import EnvWrapper
from environments.MazeMilestoneGenerator import MazeMilestonesGenerator
from environments.Milestone import PassingMilestone, ExactMilestone
from environments.gym_maze import *
from environments.gym_maze.envs import MazeEnv


def main():
    """Example for running the AdaptiveDQN"""

    # Set seed for reproducibility
    seed = 123
    # Start learning from the 0th timestep
    learning_starts = 0

    # Create some example milestones
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
        PassingMilestone(12, np.array([0.5, np.nan])),  # Milestone for reaching a position of -0.4

        # PassingMilestone(100, np.array([-0.5, np.nan]))  # Milestone for reaching a position of -0.4
    ]

    # env = EnvWrapper(gym.make("MountainCar-v0"), milestones)
    env = gym.make("maze-random-5x5-v0")
    num_milestones = 4
    milestone_generator = MazeMilestonesGenerator(env)
    milestones = milestone_generator.calculate_milestones(num_milestones)

    env = EnvWrapper(env, milestones)
    model = AdaptiveDQN(env, "MlpPolicy", env, decay_func=lambda x: np.sqrt(np.sqrt(x)), verbose=1, learning_starts=learning_starts, seed=seed)
    # model = DQN("MlpPolicy", env, verbose=1, learning_starts=learning_starts, seed=seed)
    model.learn(total_timesteps=10000)

    try:
        print("Enable render")
        env.env.env.do_enable_render()
    except AttributeError:
        pass

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        time.sleep(0.5)


    env.close()


if __name__ == '__main__':
    main()
