from operator import pos
import os
import sys
from functools import partial
import csv
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
N_RUNS = 100
FILE = "results/model_pred_noobstacles.csv"


def import_csv(csvfilename):
    data = []
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=',')
        for row in reader:
            if row:  # avoid blank lines
                data.append(row)
    return data


def sample_goals(obstacles):
    # sample 2 goals
    def sample_pos():
        return (np.random.choice(int(dims['grid_width'])),
                np.random.choice(int(dims['grid_height'])))

    if not obstacles:
        return Goal('A', sample_pos()), Goal('B', sample_pos())
    else:
        pos_temp = obstacles[0]
        while true_goal_pos := pos_temp in obstacles:
            true_goal_pos = sample_pos()
        while distr_pos := pos_temp in obstacles:
            distr_pos = sample_pos()
        return Goal('A', true_goal_pos), Goal('B', distr_pos)


def sample_obstacles():
    # TODO
    # sample obstacles
    obstacles = [(1, 1), (2, 1)]
    return []
    #return obstacles


def sample_start(dims, goals, obstacles):
    # sample starting position
    start_pos = goals[0].pos
    while start_pos in [g.pos for g in goals] + obstacles:
        start_pos = (np.random.choice(
            int(dims['grid_width'])), np.random.choice(int(dims['grid_height'])))
    return start_pos


def model_name(model_cons):
    return "{}_{}_{}".format(
        model_cons.keywords['state_type'],
        model_cons.keywords['solver'],
        model_cons.keywords['determinism'])


def evaluation(obstacles, goals, start, correct_model_idx, models_const):
    # sample obstacles, goals and start if callable

    # contruct models
    #print("constructing models")
    models = [cons(goals, obstacles, start)
                for cons in models_const]
    # choose a model
    # sample a trajectory
    #print("sampling trajectory")
    true_goal = goals[0]
    sample_trajectory = models[correct_model_idx].sample(true_goal)

    # calculate model likelihoods for all models
    #print("calculating model posteriors")
    mp = model_posteriors(sample_trajectory, true_goal, models)
    return str(obstacles)[1:-1], true_goal.pos, goals[1].pos,\
        start, model_name(model_constructors[correct_model_idx]),\
        str([s.get_coords() for s in sample_trajectory])[1:-1], mp


def sample(writer, obstacles, goals, start, correct_model_idx, model_constructors, run_idx):

    for n in tqdm(range(N_RUNS)):
        obstacles_ = obstacles() if callable(obstacles) else obstacles
        goals_ = goals(obstacles_) if callable(goals) else goals
        start_ = start(dims, goals_, obstacles_) if callable(start) else start
        correct_model_idx_ = correct_model_idx() if callable(correct_model_idx) else correct_model_idx

        obs_str, true_goal, distr, start_str, true_model, traj_str, mp = \
            evaluation(obstacles_, goals_, start_, correct_model_idx_, model_constructors)

        for t, likelihoods in enumerate(mp):
            writer.writerow([n + run_idx, dims['grid_width'], dims['grid_height'],
                            obs_str, true_goal, distr,
                            start_str, true_model, traj_str,
                            t] + list(likelihoods))


# Choose Environment
dims = {
    'shape': 'Rectangle',
    'center_x': 10,
    'center_y': 10,
    'grid_width': 20.0,
    'grid_height': 20.0}


# create all possible models
model_constructors = []
for state_type in ["GPSState", "DistanceDirectionState"]:
    for solver in ["Optimal", "Greedy"]:
        for determinism in list(np.linspace(0, 1, 3)) + [2.0, 3.0, 5.0]:
            model_constructors.append(
                partial(
                    SamplingModel,
                    dims,
                    state_type=state_type,
                    solver=solver,
                    determinism=determinism)
            )

print("number of models:", len(model_constructors))
model_names = [
    f"p_{model_name(model)}" for model in model_constructors]

#sample(writer, [], sample_goals, sample_start, model_constructors)
concave = [(6,16),(7,16),(8,16),(9,16),(9,15),(10,15),(10,14),(11,14),(11,13),(12,13),
    (12,12),(13,12),(13,11),(14,11),(14,10),(15,10),(15,9),(16,9),(16,8),(16,7),(16,6)]

convex = [
    (15,6), (16,6), (14,6), (13,6), (13,7), (12,7), (12,8), (11,8), (11,9), (10,9),
    (6,15), (6,16), (6,14), (6,13), (7,13), (7,12), (8,12), (8,11), (9,11), (9,10),
    (10,10)
]

no_obstacles = []

start = (0,0)
goals = [Goal('A', (17, 13)), Goal('B', (13, 17))]

new_file = True
with open(FILE, "w" if new_file else "a", newline='') as csv_file:
    writer = csv.writer(csv_file)
    first_run_idx = 0
    if new_file:
        writer.writerow(["run", "width", "height", "obstacles", "true_goal", "distractor", "start", "true_model", "trajectory", "t"] + model_names)
    else:
        first_run_idx = int(import_csv(FILE)[-1][0]) + 1 # idx of last run in file + 1
    sample(writer, no_obstacles, sample_goals, sample_start, lambda: np.random.choice(len(model_constructors)), model_constructors, run_idx=first_run_idx)
