from abc import ABC, abstractmethod
from typing import List, Optional

import gym
import numpy as np

from environments.Milestone import ExactMilestone, Milestone, PassingMilestone
from environments.gym_maze.envs import MazeEnv


class MilestoneGenerator(ABC):
    def __init__(self, env: gym.Env, config):
        self.env = env
        self.config = config
        self.reward = self.config["milestone_reward"]
        self.n = self.config["num_milestones"]

    @abstractmethod
    def get_milestones(self) -> List[Milestone]:
        pass

    def get_max_reward(self) -> Optional[float]:
        return None


class MazeMilestoneGenerator(MilestoneGenerator):
    """Copy of maze_2d_dijkstra.py but then in a class"""

    def __init__(self, env: MazeEnv, config):
        super().__init__(env, config)

        # Number of discrete states (bucket) per state dimension
        self.maze_size = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        self.num_buckets = self.maze_size  # one bucket per grid
        self.final_state = (self.maze_size[0] - 1, self.maze_size[1] - 1)
        self.actions = ["N", "S", "E", "W"]
        # Number of discrete actions
        self.num_actions = env.action_space.n
        # Bounds for each discrete state
        self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        self.total_distance = None

        self.compass = {
            "N": (0, -1),
            "E": (1, 0),
            "S": (0, 1),
            "W": (-1, 0)
        }

    def solve_dijkstra(self):
        # Reset the environment
        init_state = self.env.reset()

        # Initialize array with distances
        dist = np.full(self.maze_size, np.inf)
        # Initialize array that keeps track of visited cells
        isChecked = np.zeros(self.maze_size, dtype=bool)
        # Set the distance of the initial state to 0
        dist[tuple(init_state)] = 0

        while not isChecked[self.final_state]:
            # pick the vertex with the minimum distance
            dist_temp = dist.copy()
            dist_temp[isChecked] = np.inf
            minTup = np.asarray(np.unravel_index(dist_temp.argmin(), dist.shape))
            # update isChecked list
            isChecked[tuple(minTup)] = 1
            # update distances for all adjacent vertices
            for dir in self.actions:
                if self.env.maze_view.maze.is_open(minTup, dir):
                    neighTup = minTup + np.asarray(self.compass[dir])
                    if dist[tuple(neighTup)] > dist[tuple(minTup)] + 1:
                        dist[tuple(neighTup)] = dist[tuple(minTup)] + 1

        # Go backwards to find the shortest path
        curTup = self.final_state
        curDis = dist[self.final_state]
        shortest_path = [curTup]
        while np.any(curTup != init_state):
            for dir in self.actions:
                if self.env.maze_view.maze.is_open(curTup, dir):
                    neighTup = curTup + np.asarray(self.compass[dir])
                    if dist[tuple(neighTup)] == dist[tuple(curTup)] - 1:
                        curTup = neighTup
                        shortest_path.append(curTup)
                        curDis -= 1
        return dist, list(reversed(shortest_path))

    def get_milestones(self) -> List[Milestone]:
        dist, shortest_path = self.solve_dijkstra()
        print(shortest_path)
        self.total_distance = dist[tuple(self.final_state)]
        milestones = []
        interval = len(shortest_path[:-1]) // (self.n + 1)
        for idx in range(interval, len(shortest_path), interval):
            # milestones.append(ExactMilestone(reward=dist[tuple(state)], goal_state=state))
            milestones.append(ExactMilestone(reward=self.reward, goal_state=shortest_path[idx]))
            if len(milestones) == self.n:
                break
        return milestones

    def get_max_reward(self):
        return float(1 - (0.1 / self.maze_size[0]) * (self.total_distance - 1))


class MountainCarMilestoneGenerator(MilestoneGenerator):
    def __init__(self, env: gym.Env, config):
        super().__init__(env, config)

    def get_milestones(self,
                       begin_position: Optional[float] = -0.4,
                       end_position: Optional[float] = 0.5) -> List[Milestone]:
        """
        Calculate position milestones for MountainCar
        First milestone will be at begin_position
        Last milestone will be at end_position
        """
        spacing = (end_position - begin_position) / (self.n - 1)
        milestone_locations = [begin_position + i * spacing for i in range(self.n)]
        milestones = [PassingMilestone(goal_state=np.array([loc, np.nan]), reward=self.reward)
                      for i, loc in enumerate(milestone_locations)]
        return milestones
