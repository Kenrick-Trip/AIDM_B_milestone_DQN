import yaml

from experiments.Experiment import Experiment
from environments.EnvWrapper import EnvWrapper
from environments.MilestoneGenerator import MazeMilestoneGenerator
from datetime import datetime
import os
import shutil


class MazeExperiment(Experiment):
    def __init__(self, config, result_dir):
        super(MazeExperiment, self).__init__(config, result_dir)

    def get_env_wrapper(self, env, exploration_method_class):
        milestone_generator = MazeMilestoneGenerator(env, self.config)
        milestones = milestone_generator.get_milestones()
        self.config["max_reward"] = milestone_generator.get_max_reward()
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
    config, config_file = MazeExperiment.get_config_from_args(default_file="maze.yaml")

    results_folder = config['results_folder'] if 'results_folder' in config else 'experiment_no_name' # please specify folder name of results
    result_dir = os.path.join(base_dir, "results/" + results_folder, datetime.now().strftime("%Y-%m-%d-t-%H%M%S"))
    os.makedirs(result_dir)
    shutil.copy(config_file, os.path.join(result_dir, "config.yaml"))
    MazeExperiment(config, result_dir).main()

    # Check if the total distance file is there; if yes, delete it
    min_distance_path = result_dir + "/tot_dist.tmp"
    if os.path.isfile(min_distance_path):
        os.remove(result_dir + "/tot_dist.tmp")
