import time
import gym
import numpy as np
import yaml
from DQN.AdaptiveDQN import AdaptiveDQN, ExplorationMethod
from DQN.uncertainty import CountUncertainty
from stable_baselines3.common.logger import configure
import argparse

from environments.EnvWrapper import EnvWrapper


class Experiment:
    def __init__(self, config: dict, results_dir: str):
        """
        Basic Experiment template, you should extent this class for your own experiment, and
        implement the abstract method(s).
        :param config: dictionary containing the (hyper)parameters
        :param results_dir: directory to save (intermediate) results to
        """
        self.results_dir = results_dir
        self.config = config
        self.logger = configure(self.results_dir, ["log", "csv", "stdout"])

    def get_env(self):
        return gym.make(self.config['env'])

    def get_env_wrapper(self, env):
        raise NotImplementedError("You should implement the wrapper in a subclass!")

    def _train(self, model):
        model.env_wrapper.total_reset()
        model.set_logger(self.logger)
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

        exploration_method = ExplorationMethod(self.config['exploration_method'])
        if exploration_method in [ExplorationMethod.TRADITIONAL, ExplorationMethod.DEEP_EXPLORATION]:
            # If we're doing traditional epsilon greedy or deep exploration we don't need milestones
            env = EnvWrapper(self.get_env(), [])
        else:
            env = self.get_env_wrapper(self.get_env())
        uncertainty = CountUncertainty(env, **self.config[
            'uncertainty_kwargs']) if 'uncertainty_kwargs' in self.config else None

        model = AdaptiveDQN(env, self.config['policy'], env, results_folder=self.results_dir,
                            exploration_method=exploration_method, plot=self.config['plot'],
                            decay_func=lambda x: np.sqrt(x), verbose=1, learning_starts=learning_starts,
                            seed=seed, plot_update_interval=self.config["plot_update_interval"],
                            reset_heat_map_every_plot=self.config["reset_heat_map_every_plot"],
                            policy_kwargs=self.config['policy_kwargs'], uncertainty=uncertainty)

        self._train(model)
        self._demo(env, model)

    @staticmethod
    def _read_yaml(file: str, absolute=False):
        if not absolute:
            file = __file__ + "/" + file
            print(file)
        with open(file, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_config_from_args(default_file="maze.yaml"):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help="Config file to use. Other settings will override the config "
                                                   "that is read in from the file", metavar="FILE",
                            default=default_file)
        args = parser.parse_args()
        config = Experiment._read_yaml(args.config, absolute=True if args.config != default_file else True)

        print("Using config file ", args.config, "!")
        return config, args.config
