import os
from environments.gym_maze.envs.maze_view_2d import Maze

if __name__ == "__main__":

    """
    Steps to generate and load a new maze:
    1. Change the dimensionality of the maze to create!
    2. Go to "maze_env.py" and create a new class, which loads the generated file
    3. Go to "gym_maze/__init__.py" and create a new registry for the created class
    4. To run the environment, load the newly created environment "maze_2d_q_learning.py"
    """
    maze_dim = 61
    maze = Maze(maze_size=(maze_dim, maze_dim))

    # check if the folder "maze_samples" exists in the current working directory
    dir_name = os.path.join(os.getcwd(), "maze_samples")
    if not os.path.exists(dir_name):
        # create it if it doesn't
        os.mkdir(dir_name)

    # increment number until it finds a name that is not being used already (max maze_999)
    maze_path = None
    for i in range(1, 1000):
        maze_name = "maze2d_{}x{}_{}.npy".format(maze_dim, maze_dim, str(i).zfill(3))
        maze_path = os.path.join(dir_name, maze_name)
        if not os.path.exists(maze_path):
            break
        if i == 999:
            raise ValueError("There are already 999 mazes in the %s." % dir_name)


    maze.save_maze(maze_path)
    print("New maze generated and saved at %s." %  maze_path)

