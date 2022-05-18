from experiments.Experiment import Experiment
from environments.EnvWrapper import EnvWrapper
from environments.MazeMilestoneGenerator import MazeMilestoneGenerator
from datetime import datetime
import os
import shutil


class MazeExperiment(Experiment):
    def get_env_wrapper(self, env):
        milestone_generator = MazeMilestoneGenerator(env)
        milestones = milestone_generator.get_milestones(self.config["num_milestones"])
        return EnvWrapper(env, milestones)


if __name__ == '__main__':
    config, config_file = MazeExperiment.get_config_from_args(default_file="maze.yaml")
    result_dir = "./results/" + datetime.now().strftime("%Y-%m-%d-t-%H:%M:%S")
    os.makedirs(result_dir)
    shutil.copy(config_file, result_dir + "/config.yaml")
    MazeExperiment(config, result_dir).main()
