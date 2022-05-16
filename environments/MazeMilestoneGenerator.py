from typing import List

import gym
import numpy as np

from environments.Milestone import ExactMilestone
from environments.gym_maze.envs import MazeEnv


class MazeMilestonesGenerator:
    """Copy of maze_2d_dijkstra.py but then in a class"""
    def __init__(self, env: MazeEnv):
        self.env = env

        # Number of discrete states (bucket) per state dimension
        self.maze_size = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        self.num_buckets = self.maze_size  # one bucket per grid
        self.final_state = (self.maze_size[0] - 1, self.maze_size[1] - 1)
        self.actions = ["N", "S", "E", "W"]
        # Number of discrete actions
        self.num_actions = env.action_space.n
        # Bounds for each discrete state
        self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

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
        return dist, shortest_path

    def calculate_milestones(self, num_milestones: int) -> List[ExactMilestone]:
        dist, shortest_path = self.solve_dijkstra()
        tot_dist = dist[tuple(self.final_state)]

        milestones = []
        for state in shortest_path[1:-1]:
            if dist[tuple(state)] <= tot_dist * (num_milestones - len(milestones) - 1) / num_milestones:
                milestones.append(ExactMilestone(reward=dist[tuple(state)], goal_state=state))
            if len(milestones) == num_milestones:
                break
        return milestones
