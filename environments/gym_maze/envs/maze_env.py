import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from environments.gym_maze.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True):

        self.viewer = None

        # Force it to be false for now
        enable_render = False
        self.enable_render = enable_render

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size "
                                 "(tuple of length 2)")

        self.maze_size = self.maze_view.maze_size
        self.init_state = np.asarray(self.maze_size) // 2

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high = np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int) or isinstance(action, np.int64) or isinstance(action, np.int32):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(self.ACTION[int(action)])

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            done = True
        else:
            reward = -0.1/self.maze_size[0]
            #reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

        self.state = self.maze_view.robot
        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.maze_view.reset_robot()
        # self.state = np.zeros(2)
        self.state = self.init_state
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)

    def render_shortest_path(self, shortest_path, mode="human", close=False, timestamp=None):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update_shortest_path(shortest_path, mode, timestamp=timestamp)

    def render_milestones(self, milestones, mode="human", close=False, timestamp=None):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update_milestones(milestones, mode, timestamp=timestamp)

    def do_enable_render(self):
        """Enable rendering, start up pygame"""
        if self.enable_render:
            return

        self.enable_render = True
        self.maze_view.do_enable_render()


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), enable_render=enable_render)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", enable_render=enable_render)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), enable_render=enable_render)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", enable_render=enable_render)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), enable_render=enable_render)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", enable_render=enable_render)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), enable_render=enable_render)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", enable_render=enable_render)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, enable_render=True):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", enable_render=enable_render)

class MazeEnvSample11x11(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample11x11, self).__init__(maze_file="maze2d_11x11.npy",
                                                  enable_render=enable_render)

class MazeEnvSample101x101(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample101x101, self).__init__(maze_file="maze2d_101x101.npy",
                                                   enable_render=enable_render)

class MazeEnvSample21x21(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample21x21, self).__init__(maze_file="maze2d_21x21.npy",
                                                   enable_render=enable_render)


class MazeEnvSample15x15(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample15x15, self).__init__(maze_file="maze2d_15x15.npy",
                                                 enable_render=enable_render)

class MazeEnvCustom10x10(MazeEnv):
    def __init__(self, enable_render=True):
        super(MazeEnvCustom10x10, self).__init__(maze_file="maze2d_10x10_custom.npy",
                                                  enable_render=enable_render)
