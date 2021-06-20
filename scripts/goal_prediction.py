import os
import sys
from functools import partial
from matplotlib.pyplot import plot
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath('./src/'))
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pprint import pprint

from src.calcs.posteriors import model_posteriors
from src.data_tools.utils import validate_inputs
from src.glb import SETTINGS
from src.model import Goal, SamplingModel
from src.visualization import Visualizer, plotPosteriors

SETTINGS.norm = "L2"


# TODO start with goals as only difference?
# TODO Which model is more likely?
# TODO do it over multiple samples

# USER SPECIFIED PREFERENCES:
visualize = False
N_RUNS = 2


def sample_goals():
    # sample 2 goals
    true_goal = Goal('A', (np.random.choice(dims['grid_width']), np.random.choice(dims['grid_height'])))
    distractor = Goal('B', (np.random.choice(dims['grid_width']), np.random.choice(dims['grid_height'])))
    return [true_goal, distractor]


def sample_obstacles():
    # sample obstacles
    obstacles = [(1, 1), (2, 1)]
    return obstacles


def sample_start(dims, goals, obstacles):
    # sample starting position
    start_pos = goals[0].pos
    while start_pos in [g.pos for g in goals] + obstacles:
        start_pos = (np.random.choice(dims['grid_width']), np.random.choice(dims['grid_height']))
    return start_pos


# Choose Environment
dims = {
    'shape': 'Rectangle',
    'center_x': 5,
    'center_y': 5,
    'grid_width': 10,
    'grid_height': 10}


# create all possible models
models = []
for state_type in ["GPSState", "DistanceDirectionState"]:
    for solver in ["Optimal", "Greedy"]:
        for determinism in np.arange(4.5, 5, .25):
            models.append(
                partial(
                    SamplingModel,
                    dims,
                    state_type=state_type,
                    solver=solver,
                    determinism=determinism)
            )


# for every model, do evaluation
for model_constructor in models:
    print(model_constructor.keywords['state_type'],
          model_constructor.keywords['solver'],
          model_constructor.keywords['determinism'])
    for n in tqdm(range(N_RUNS)):
        goals = sample_goals()
        true_goal = goals[0]
        obstacles = sample_obstacles()
        start = sample_start(dims, goals, obstacles)

        model = model_constructor(goals, obstacles, start)

        viz = Visualizer(model)

        # calculate optimal policies
        policies = model.get_policies()

        #viz.policy(true_goal)

        sample_trajectory = model.sample(true_goal)
        #viz.trajectory(true_goal, sample_trajectory)

        posteriors = model.get_posteriors(policies, sample_trajectory)
        plotPosteriors(posteriors)
