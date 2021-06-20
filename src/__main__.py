"""
https://github.com/stacyste/TheoryOfMindInferenceModels
"""

import math
from pprint import pprint

import click
from matplotlib.patches import Circle

from src.calcs import posteriors
from src.calcs.mdp import TenenbaumMDPSolver
from src.data_tools import io
from src.glb import SETTINGS
from src import model
from src import visualization


# TODO
# TODO calculate reward table instead of calc
# TODO where to move solvers?
# TODO model from file implementation (maybe instead of datamodel just do a static from_file?)
# TODO MDP and MDPSolver
# TODO move beta to solver class
# TODO make get_boltzman private

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


def visualize(model_, goal):
    print("visualization")
    viz = visualization.Visualizer(model_)
    print("visualize environment")
    viz.show_environment()
    if len(model_.env.coords) <= 10:
        print("visualize reward table")
        viz.reward_table(goal)
    print("visualize optimal values")
    viz.optimal_values(goal)
    if len(model_.env.coords) <= 10:
        print("visualize policy")
        viz.policy(goal)


def get_posteriors(policies, priors, trajectory):
    return posteriors.get_posteriors(
        policies, priors, trajectory)


def get_policies(mdp_solver):
    return {goal: mdp_solver.policies(goal.pos)
            for goal in mdp_solver.mdp.goals}


def main(arena_dims, goals, traps, data, trajectory, viz=False):
    #SETTINGS.state_type = "DistanceDirectionState"

    print("build MDP")
    agent = model.Agent()
    env = model.Environment(arena_dims, goals, traps)
    print(env.coords)

    model_ = model.Model(env)
    mdp_solver = TenenbaumMDPSolver(model_)

    # Generate the reward tables for each goal
    # Calculate policies
    # goal object: state: action : value
    print("calculating policies")
    pi_policies = model_.get_policies()  # get_policies(mdp_solver)
    print("done.")
    print("policies")
    pprint(pi_policies)

    # TODO vis trajectory

    if viz:
        visualize(model_, env.get_goal(viz['goal']))

    goal_priors = [.5, .5]
    assert sum(goal_priors) == 1

    # set the trajectory
    trajectory = list(map(model_.mdp.get_state, trajectory))

    """
    gi = posteriors.PerformGoalInference(mdp.T, pi_policies, [.5,.5], trajectory)
    posts = gi()
    print(posts)
    return posts
    """

    print("calculate posteriors")
    norm_posteriors = get_posteriors(pi_policies, goal_priors, trajectory)

    print("\nPOSTERIORS\n", norm_posteriors)
    if viz:
        visualization.plotPosteriors(
            norm_posteriors, f"Posteriors to {viz['goal']}")
    return norm_posteriors


def setup_args(height, width, goals, traps):
    """"""
    return goals


if __name__ == "__main__":

    # environment definition
    traps = []

    # data stuff
    true_goal = 'A'
    pix = 22
    data_path, file_name, goals = "data/for_paulina/data/", "data/for_paulina/data/flytrax20200910_173011.csv", {
        'A': (
            600, 500), 'B': (
            400, 200)}
    #data_path, file_name, goals, pix = "tests/data/", "tests/data/testsimple.csv", {'A': (3, 3), 'B': (3, 2)}, 1

    goals = {g: (int(math.floor(coords[0] / pix)),
                 int(math.floor(coords[1] / pix))) for g,
             coords in goals.items()}
    print("goals:", goals)

    main(goals, traps, data_path, file_name=file_name, pixelation=pix)
