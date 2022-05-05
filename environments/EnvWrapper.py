import gym


class EnvWrapper:
    def __init__(self, env: gym.Env):
        self.env = env

    def step(self, action):
        return self.env.step(action=action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def spec(self):
        return self.env.spec

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __str__(self):
        return self.env.__str__()

    def __enter__(self):
        """Support with-statement for the environment."""
        return self.env.__enter__()

    def __exit__(self, *args):
        return self.env.__exit__()
