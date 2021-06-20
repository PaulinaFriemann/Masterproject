import os
import sys

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
visualize = True

# Choose Environment
dims = {
    'shape': 'Rectangle',
    'center_x': 10,
    'center_y': 10,
    'grid_width': 20.0,
    'grid_height': 20.0}
goalA = Goal('A', (13, 17))
goalB = Goal('B', (17, 13))
goals = [goalA, goalB]  # {'A': (3, 3), 'B': (3, 1)}
# [(1,3), (2,3), (3,3), (4,3), (5,3), (5,2), (5,1)]

obstacles = [(18,5), (18,6), (18,7), (18,8), (17,8),
             (17,9),(17,10), (16,10), (16,11), (15,11), (15,12), (14,12), (13,13), (14,13),
             (5,18), (6,18), (7,18), (8,18), (8,17),
             (9,17), (10,17), (10,16), (11,16), (11,15), (12,15), (12,14), (13,14)]



obstacles = [
    (15,6), (16,6), (14,6), (13,6), (13,7), (12,7), (12,8), (11,8), (11,9), (10,9),
    (6,15), (6,16), (6,14), (6,13), (7,13), (7,12), (8,12), (8,11), (9,11), (9,10),
    (10,10)
]


obstacles_ = [(6,16),(7,16),(8,16),(9,16),(9,15),(10,15),(10,14),(11,14),(11,13),(12,13),
    (12,12),(13,12),(13,11),(14,11),(14,10),(15,10),(15,9),(16,9),(16,8),(16,7),(16,6)]

# goal_priors

# Change to reflect true goal/environment of choice
start_pos = (0, 0)  # any state in the world

# model 1: GPS state

model1 = SamplingModel(
    dims,
    goals,
    obstacles,
    start_pos,
    state_type="DistanceDirectionState",
    solver="Greedy",
    determinism=1.0)

# model 2: Distance Direction state

model2 = SamplingModel(
    dims,
    goals,
    obstacles,
    start_pos,
    state_type="GPSState",
    determinism=3.0)
#validate_inputs(goals, [model1, model2], start_pos)

policies1 = model1.get_policies()
#policies2 = model2.get_policies()

true_goal = goalB

# visualization
if True:
    for goal in goals:
        viz = Visualizer(model2)
        viz.show_environment()
        exit()
        # viz.reward_table(goal)
        viz.optimal_values(true_goal)
        # viz.policy(goal)

if False:
    for goal in goals:
        viz = Visualizer(model1)
        # viz.show_environment()
        # viz.reward_table(goal)
        viz.optimal_values(true_goal)
        # viz.policy(goal)

sample_trajectory = model2.sample(true_goal)

if visualize:
    viz = Visualizer(model1)
    viz.trajectory(true_goal, sample_trajectory)

exit()

if visualize:
    posteriors1 = model1.get_posteriors(policies1, sample_trajectory)
    posteriors2 = model2.get_posteriors(policies2, sample_trajectory)
    # print(posteriors1)
    # print(posteriors2)
    plotPosteriors(
        posteriors1,
        f"Posteriors to {goal}",
        labels=[
            goal.name for goal in goals])
    plotPosteriors(
        posteriors2,
        f"Posteriors to {goal}",
        labels=[
            goal.name for goal in goals])

model1.get_posteriors(policies1, sample_trajectory)
model_posteriors(sample_trajectory, true_goal, [model1, model2])
