from experiments.Experiment import Experiment
from environments.EnvWrapper import EnvWrapper
from environments.MazeMilestoneGenerator import MountainCarMilestoneGenerator
from datetime import datetime
import os
import shutil


class MountainCarExperiment(Experiment):
    def get_env_wrapper(self, env):
        milestone_generator = MountainCarMilestoneGenerator(env)
        milestones = milestone_generator.get_milestones(self.config["num_milestones"])
        return EnvWrapper(env, milestones)


if __name__ == '__main__':
    config, config_file = MountainCarExperiment.get_config_from_args(default_file="mountaincar.yaml")
    result_dir = "./results/" + datetime.now().strftime("%Y-%m-%d-t-%H:%M:%S")
    os.makedirs(result_dir)
    shutil.copy(config_file, result_dir + "/config.yaml")
    MountainCarExperiment(config, result_dir).main()
