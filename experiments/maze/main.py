from experiments.Experiment import Experiment
from environments.EnvWrapper import EnvWrapper
from environments.MilestoneGenerator import MazeMilestoneGenerator
from datetime import datetime
import os
import shutil


class MazeExperiment(Experiment):
    def __init__(self, config, result_dir):
        super(MazeExperiment, self).__init__(config, result_dir)
    def get_env_wrapper(self, env):
        milestone_generator = MazeMilestoneGenerator(env, self.config["milestone_reward"])
        milestones = milestone_generator.get_milestones(self.config["num_milestones"], self.results_dir)
        print(milestones)
        return EnvWrapper(env, milestones)


if __name__ == '__main__':
    config, config_file = MazeExperiment.get_config_from_args(default_file="maze.yaml")
    result_dir = "./results_maze/" + datetime.now().strftime("%Y-%m-%d-t-%H%M%S")
    os.makedirs(result_dir)
    shutil.copy(config_file, result_dir + "/config.yaml")
    MazeExperiment(config, result_dir).main()

    # Check if the total distance file is there; if yes, delete it
    min_distance_path = result_dir +"/tot_dist.tmp"
    if os.path.isfile(min_distance_path):
        os.remove(result_dir+"/tot_dist.tmp")
