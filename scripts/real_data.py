import os
import sys

sys.path.append(os.path.abspath('./src/'))
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#from src.__main__ import main
import math
import numpy as np
from src.data_tools import io
from src.model import Environment, Goal, DataModel
from src.visualization import Visualizer, plotPosteriors


def load_data(data_path, file_name=None, pixelation=1, test_phase=False):
    # load the data
    print("loading the data: ", data_path, file_name)
    reader = io.CSVReader(data_path, pixelation=pixelation)
    if not file_name:
        file_name = reader.files[0]
    if test_phase:
        df = reader.test_phase(file_name)
    else:
        df = reader._single_file(file_name)
    trajectory = reader.trajectory(df)
    arena_dims = reader.dimensions(file_name)

    # Generate the environment with graph for every goal
    return df, arena_dims, trajectory


viz = True

# environment definition
obstacles = []

# data stuff
true_goal = 'A'

pix = 2


data_path, file_name, goals = "tests/data/", "tests/data/example2.csv", {
    'A': (
        600, 500), 'B': (
            400, 200)}
goalA = Goal('A', (6, 4))
goalB = Goal('B', (5, 7))
goals = [goalA, goalB]

dm = DataModel("tests/data/example2.csv", goals, obstacles, pixelation=pix)
goals = dm.goals
goalA = goals[0]
goalB = goals[1]
obstacles = dm.obstacles


viz = Visualizer(dm)
#viz.show_environment()
#viz.reward_table(goalA)

policies = dm.get_policies()
#viz.optimal_values(goalA)
#viz.policy(goalA)

posteriors = dm.get_posteriors(policies, dm.trajectory())
viz.trajectory(goalA, dm.trajectory())
plotPosteriors(posteriors)
