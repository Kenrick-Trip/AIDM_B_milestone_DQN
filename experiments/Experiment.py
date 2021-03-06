import os
import sys
import time
import gym
import numpy as np
import yaml
from DQN.AdaptiveDQN import AdaptiveDQN, ExplorationMethod
from stable_baselines3.common.callbacks import EvalCallback
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

    def get_env_wrapper(self, env, exploration_method_class):
        raise NotImplementedError("You should implement the wrapper in a subclass!")

    def _train(self, env, model):
        eval_callback = EvalCallback(env, log_path=self.results_dir, eval_freq=self.config["eval_rate"],
                                     deterministic=True, render=False)
        model.env_wrapper.total_reset()
        model.set_logger(self.logger)
        model.learn(total_timesteps=self.config["trainsteps"], callback=eval_callback)

    @staticmethod
    def _prepare_demo(env):
        """Make sure environment is ready for demo"""
        # We have to force enable render for maze
        # TODO: there must be a cleaner way !
        try:
            env.env.env.do_enable_render()
        except AttributeError:
            pass

    def _demo(self, env, model):
        self._prepare_demo(env)
        obs = env.reset()
        for i in range(self.config["demosteps"]):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
            time.sleep(0.5)

    def main(self):
        """Example for running the AdaptiveDQN"""

        # Set seed for reproducibility
        if 'seed' not in self.config:
            seed = np.random.randint(10000)
            self.config['seed'] = seed
        else:
            seed = self.config['seed']
        # Start learning from the 0th timestep
        learning_starts = 0

        self.exploration_method = ExplorationMethod(self.config['exploration_method'])
        env = self.get_env_wrapper(self.get_env(), ExplorationMethod)

        # Special mode for quick view of environment
        if self.config['visualize_environment_only']:
            self._prepare_demo(env)
            self._visualize_env_and_exit(env)

        uncertainty = CountUncertainty(env, **self.config[
            'uncertainty_kwargs']) if 'uncertainty_kwargs' in self.config else None

        model = AdaptiveDQN(env, self.config['policy'], env, results_folder=self.results_dir,
                            exploration_method=self.exploration_method, config=self.config,
                            decay_func=lambda x: np.sqrt(x), verbose=1, learning_starts=learning_starts,
                            seed=seed, policy_kwargs=self.config['policy_kwargs'], uncertainty=uncertainty,
                            exploration_fraction=self.config['exploration_fraction'],
                            learning_rate=self.config['learning_rate'],
                            max_reward=self.config.get("max_reward"),
                            gradient_steps=self.config.get("gradient_steps"), train_freq=(1, 'episode'),
                            batch_size=self.config.get("batch_size"),
                            buffer_size=self.config.get("buffer_size"))

        self._train(env, model)
        # self._demo(env, model)

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

    @staticmethod
    def _visualize_env_and_exit(env):
        env.render()
        input("Press any key to close...")
        sys.exit(0)
