from experiments.Experiment import Experiment
from environments.EnvWrapper import EnvWrapper
from environments.MilestoneGenerator import MountainCarMilestoneGenerator
from datetime import datetime
import os
import shutil


class MountainCarExperiment(Experiment):
    def get_env_wrapper(self, env):
        milestone_generator = MountainCarMilestoneGenerator(env, self.config)
        milestones = milestone_generator.get_milestones()
        return EnvWrapper(env, milestones)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    config, config_file = MountainCarExperiment.get_config_from_args(default_file="mountaincar.yaml")
    result_dir = os.path.join(base_dir, "results_mc", datetime.now().strftime("%Y-%m-%d-t-%H%M%S"))
    os.makedirs(result_dir)
    shutil.copy(config_file, os.path.join(result_dir, "config.yaml"))
    MountainCarExperiment(config, result_dir).main()
