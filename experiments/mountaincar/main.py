import yaml

from experiments.Experiment import Experiment
from environments.EnvWrapper import EnvWrapper
from environments.MilestoneGenerator import MountainCarMilestoneGenerator
from datetime import datetime
import os
import shutil


class MountainCarExperiment(Experiment):
    def get_env_wrapper(self, env, exploration_method_class):
        milestone_generator = MountainCarMilestoneGenerator(env, self.config)
        milestones = milestone_generator.get_milestones()
        self.config["max_reward"] = 1 # only for the "MountainCar-v0" env
        yaml.safe_dump(self.config, open(os.path.join(self.results_dir, "config.yaml"), "w"))
        if self.exploration_method in [exploration_method_class.TRADITIONAL,
                                       exploration_method_class.DEEP_EXPLORATION]:
            # If we're doing traditional epsilon greedy or deep exploration we don't need milestones
            return EnvWrapper(self.get_env(), [])
        else:
            print(milestones)
            return EnvWrapper(env, milestones)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    config, config_file = MountainCarExperiment.get_config_from_args(default_file="mountaincar.yaml")
    result_dir = os.path.join(base_dir, "results_mc_run_bas1", datetime.now().strftime("%Y-%m-%d-t-%H%M%S"))
    os.makedirs(result_dir)
    shutil.copy(config_file, os.path.join(result_dir, "config.yaml"))
    MountainCarExperiment(config, result_dir).main()
