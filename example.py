import gym
import numpy as np

from DQN.AdaptiveDQN import AdaptiveDQN
from environments.EnvWrapper import EnvWrapper
from environments.Milestone import PassingMilestone


def main():
    """Example for running the AdaptiveDQN"""

    # Set seed for reproducibility
    seed = 123
    # Start learning from the 0th timestep
    learning_starts = 0

    # Create some example milestones
    milestones = [
        PassingMilestone(10, np.array([np.nan, 0.01])),  # Milestone for reaching a velocity of 0.01
        PassingMilestone(100, np.array([-0.4, np.nan]))  # Milestone for reaching a position of -0.4
    ]

    env = EnvWrapper(gym.make("MountainCar-v0"), milestones)
    model = AdaptiveDQN("MlpPolicy", env, verbose=1, learning_starts=learning_starts, seed=seed)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
