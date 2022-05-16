import sys
import numpy as np
from datetime import datetime
from pathlib import Path

import gym
import environments.gym_maze

def solve_dijkstra():

    COMPASS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    # Render tha maze
    env.render()

    # Reset the environment
    init_state = env.reset()


    # Initialize array with distances
    dist = np.full(MAZE_SIZE, np.inf)
    # Initialize array that keeps track of visited cells
    isChecked = np.zeros(MAZE_SIZE, dtype=bool)
    # Set the distance of the initial state to 0
    dist[tuple(init_state)] = 0

    while not isChecked[FINAL_STATE]:
        # pick the vertex with the minimum distance
        dist_temp = dist.copy()
        dist_temp[isChecked] = np.inf
        minTup = np.asarray(np.unravel_index(dist_temp.argmin(), dist.shape))
        # update isChecked list
        isChecked[tuple(minTup)] = 1
        # update distances for all adjacent vertices
        for dir in ACTIONS:
            if env.maze_view.maze.is_open(minTup, dir):
                neighTup = minTup + np.asarray(COMPASS[dir])
                if dist[tuple(neighTup)] > dist[tuple(minTup)] + 1:
                    dist[tuple(neighTup)] = dist[tuple(minTup)] + 1

    # Go backwards to find the shortest path
    curTup = FINAL_STATE
    curDis = dist[FINAL_STATE]
    shortest_path = [curTup]
    while np.any(curTup != init_state):
        for dir in ACTIONS:
            if env.maze_view.maze.is_open(curTup, dir):
                neighTup = curTup + np.asarray(COMPASS[dir])
                if dist[tuple(neighTup)] == dist[tuple(curTup)] - 1:
                    curTup = neighTup
                    shortest_path.append(curTup)
                    curDis -= 1

    # Render the shortest path
    env.render_shortest_path(shortest_path, timestamp=timestamp)

    return dist, shortest_path

def solve_dijkstra_no_render(env):
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
    FINAL_STATE = (MAZE_SIZE[0]-1, MAZE_SIZE[1]-1)
    ACTIONS = ["N", "S", "E", "W"]

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    COMPASS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }
    # Reset the environment
    init_state = env.reset()

    # Initialize array with distances
    dist = np.full(MAZE_SIZE, np.inf)
    # Initialize array that keeps track of visited cells
    isChecked = np.zeros(MAZE_SIZE, dtype=bool)
    # Set the distance of the initial state to 0
    dist[tuple(init_state)] = 0

    while not isChecked[FINAL_STATE]:
        # pick the vertex with the minimum distance
        dist_temp = dist.copy()
        dist_temp[isChecked] = np.inf
        minTup = np.asarray(np.unravel_index(dist_temp.argmin(), dist.shape))
        # update isChecked list
        isChecked[tuple(minTup)] = 1
        # update distances for all adjacent vertices
        for dir in ACTIONS:
            if env.maze_view.maze.is_open(minTup, dir):
                neighTup = minTup + np.asarray(COMPASS[dir])
                if dist[tuple(neighTup)] > dist[tuple(minTup)] + 1:
                    dist[tuple(neighTup)] = dist[tuple(minTup)] + 1

    # Go backwards to find the shortest path
    curTup = FINAL_STATE
    curDis = dist[FINAL_STATE]
    shortest_path = [curTup]
    while np.any(curTup != init_state):
        for dir in ACTIONS:
            if env.maze_view.maze.is_open(curTup, dir):
                neighTup = curTup + np.asarray(COMPASS[dir])
                if dist[tuple(neighTup)] == dist[tuple(curTup)] - 1:
                    curTup = neighTup
                    shortest_path.append(curTup)
                    curDis -= 1
    return dist, shortest_path

def calculate_milestones(num_milestones, dist, shortest_path):
    tot_dist = dist[tuple(FINAL_STATE)]
    # Initialize milestones and rewards lists
    m_states = []
    m_rewards = []

    for state in shortest_path[1:-1]:
        if dist[tuple(state)] <= tot_dist * (num_milestones - len(m_states) - 1) / num_milestones:
            m_states.append(state)
            m_rewards.append(dist[tuple(state)])
        if len(m_states) == num_milestones:
            break

    env.render_milestones(m_states, timestamp=timestamp)
    return m_states, m_rewards

def calculate_milestones_no_render(env, num_milestones, dist, shortest_path):
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
    FINAL_STATE = (MAZE_SIZE[0]-1, MAZE_SIZE[1]-1)
    ACTIONS = ["N", "S", "E", "W"]

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    tot_dist = dist[tuple(FINAL_STATE)]
    # Initialize milestones and rewards lists
    m_states = []
    m_rewards = []

    for state in shortest_path[1:-1]:
        if dist[tuple(state)] <= tot_dist * (num_milestones - len(m_states) - 1) / num_milestones:
            m_states.append(state)
            m_rewards.append(dist[tuple(state)])
        if len(m_states) == num_milestones:
            break
    return m_states, m_rewards


if __name__ == "__main__":

    # Create a folder to save everything
    Path("output_files/").mkdir(parents=True, exist_ok=True)
    # Get a timestamp for execution
    global timestamp
    timestamp = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    # Initialize the "maze" environment
    # env = gym.make("maze-random-10x10-plus-v0")
    env = gym.make("maze-sample-101x101-v0")
    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
    FINAL_STATE = (MAZE_SIZE[0]-1, MAZE_SIZE[1]-1)
    ACTIONS = ["N", "S", "E", "W"]

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Defining the rendering related constants
    '''
    RENDER_MAZE = True
    ENABLE_RECORDING = False

    '''
    Begin simulation
    '''
    recording_folder = "/tmp/maze_q_learning"

    if ENABLE_RECORDING:
        env.monitor.start(recording_folder, force=True)

    # simulate()
    dist, shortest_path = solve_dijkstra()

    num_milestones = 3
    # calculate the milestones
    m_states, m_rewards = calculate_milestones(num_milestones, dist, shortest_path)

    # Save the milestones statistics
    np.savetxt("output_files/milestones" + timestamp + ".csv", m_states, delimiter=",")
    np.savetxt("output_files/milestone_rewards" + timestamp + ".csv", m_rewards, delimiter=",")

    print("Files saved under {}, with timestamp: {}".format("output_files", timestamp))
    if ENABLE_RECORDING:
        env.monitor.close()
