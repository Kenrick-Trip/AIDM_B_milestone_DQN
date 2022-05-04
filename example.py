import gym

from DQN.AdaptiveDQN import AdaptiveDQN


def main():
    """Example for running the AdaptiveDQN"""

    # Set seed for reproducibility
    seed = 123
    # Start learning from the 0th timestep
    learning_starts = 0

    env = gym.make("CartPole-v1")
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