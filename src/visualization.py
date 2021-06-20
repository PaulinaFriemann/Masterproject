import functools
import copy
import operator
import itertools as it

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from src.data_tools import utils as transformations


GRID_ADJUST = .5
GRID_SCALE = 2.5


def _rect(x, y, fill, color, alpha):
    bounds = (x - GRID_ADJUST, y - GRID_ADJUST)
    return Rectangle(bounds, 1, 1, fill=fill,
                     color=color,
                     alpha=alpha)


line = functools.partial(_rect, fill=False, color='black', alpha=.1)
outside = functools.partial(_rect, fill=True, color='black', alpha=.1)
trap = functools.partial(_rect, fill=True, color='red', alpha=.1)
goal = functools.partial(_rect, fill=True, color='green', alpha=.1)
true_goal = functools.partial(_rect, fill=True, color='green', alpha=.5)
traj = functools.partial(_rect, fill=True, color='blue', alpha=.1)


def sort_xy(positions):
    return sorted(
        sorted(positions, key=operator.itemgetter(0)),
        key=operator.itemgetter(1), reverse=True)


class Cell:
    def __init__(self, pos, type_, text=""):
        self.type_ = type_
        self.pos = pos
        self.text = text

    def draw(self, ax):
        ax.add_patch(globals()[self.type_](*self.pos))
        ax.text(*self.pos, self.text, ha="center")

    @classmethod
    def from_state(cls, state, goals, traps):
        if state in traps:
            type_ = "trap"
        elif state in [g.pos for g in goals]:
            type_ = "goal"
        else:
            type_ = "line"

        return cls(tuple(state), type_)

    def __str__(self):
        return f"{self.type_} Cell {self.pos} {self.text}"


class Grid:
    def __init__(self, lower_x, lower_y, upper_x, upper_y, title=""):
        self.cells = []
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.upper_y = upper_y
        self.title = title

    def draw(self):
        for cell in self.cells:
            cell.draw(self.ax)
        self.ax.set_title(self.title)

    def get_cell(self, position):
        for cell in self.cells:
            if cell.pos == position:
                return cell
        return Cell(position, "outside")

    @property
    def ax(self):
        return self.ax__

    @ax.setter
    def ax(self, ax):
        ax.set_xbound(lower=self.lower_x - .5, upper=self.upper_x + .5)
        ax.set_ybound(lower=self.lower_y - .5, upper=self.upper_y + .5)

        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.set_xticks(range(0, self.upper_x + 1, 5))
        ax.set_yticks(range(0, self.upper_y + 1, 5))
        self.ax__ = ax

    def add_text_from_series(self, series):
        self.title = series.name
        for action, value in series.items():
            pos = tuple(
                action[dim] +
                val for dim,
                val in enumerate(
                    series.name))
            cell = self.get_cell(pos)
            cell.text = round(value, 3)
            if cell not in self.cells:
                self.cells.append(cell)

    @classmethod
    def from_grid(cls, grid, ax):
        new_ = cls(grid.lower_x, grid.lower_y, grid.upper_x, grid.upper_y)
        new_.cells = copy.deepcopy(grid.cells)
        new_.ax = ax
        return new_

    @classmethod
    def from_df(cls, df):
        pass

    @classmethod
    def from_env(cls, env):
        minimumx, minimumy = [min(coord) for coord in zip(*env.coords)]
        maximumx, maximumy = [max(coord) for coord in zip(*env.coords)]

        inst = cls(minimumx - 1, minimumy - 1, maximumx + 1, maximumy + 1)

        # first y values inversed, then x values
        for state in sort_xy(env.coords):
            cell = Cell.from_state(state, env.goals, env.obstacles)

            # if state in [goal.pos for goal in env.goals]:
            text = [g.name for g in env.goals if state == g.pos]
            cell.text = text[0] if text else ""

            inst.cells.append(cell)

        # plt.rcParams["figure.figsize"] = [
        #    (maximumx - minimumx) * GRID_SCALE,
        #    (maximumy - minimumy) * GRID_SCALE]

        return inst

    def __str__(self):
        s = ""
        for i, cell in enumerate(self.cells):
            if not i % (self.upper_x - self.lower_x):
                s += "\n" + ("---------" *
                             (self.upper_x - self.lower_x)) + "\n"
            s += " | " + cell.type_ + ":" + cell.text
        return s


class Visualizer:
    def __init__(self, model):
        self.model = model
        self.mdp = model.mdp
        self.grid = Grid.from_env(self.mdp.env)

    def show_environment(self):
        self.grid.ax = plt.gca()
        plt.gca().set_aspect('equal', adjustable='box')
        self.grid.draw()
        plt.show()

    def draw_multiple(self, axs, df):
        for pos, ax in zip(sort_xy(self.mdp.env.coords),
                           it.chain.from_iterable(axs)):
            row = df.loc[[pos]].squeeze()

            grid = Grid.from_grid(self.grid, ax)
            grid.add_text_from_series(row)
            grid.draw()

    def reward_table(self, goal_object):
        goal_cell = self.grid.get_cell(goal_object.pos)
        goal_cell.type_ = "true_goal"

        table = self.mdp.reward_table(goal_object)

        fig, axs = plt.subplots((self.grid.upper_x - self.grid.lower_x - 1),
                                (self.grid.upper_y - self.grid.lower_y - 1),
                                subplot_kw={"frame_on": False})
        df = transformations.dict_to_dataframe(table)

        self.draw_multiple(axs, df)
        fig.suptitle(
            "Reward function for goal {}".format(
                goal_object.name),
            fontsize=16)
        plt.show()

    def optimal_values(self, goal_object):
        goal_cell = self.grid.get_cell(goal_object.pos)
        goal_cell.type_ = "true_goal"
        table = {state.get_coords(): v for state,
                 v in self.model.solver.utilities[goal_object].items()}

        table = pd.Series(table)
        table.name = (0, 0)

        __, ax = plt.subplots(subplot_kw={"frame_on": False})
        grid = Grid.from_grid(self.grid, ax)
        grid.add_text_from_series(table)
        grid.title = "Utility values for goal {}".format(goal_object.name)

        grid.draw()
        plt.show()

    def policy(self, goal_object):
        table = {state.get_coords(): v for state,
                 v in self.model.get_policies()[goal_object].items()}

        fig, axs = plt.subplots((self.grid.upper_x - self.grid.lower_x - 1),
                                (self.grid.upper_y - self.grid.lower_y - 1),
                                subplot_kw={"frame_on": False})

        df = pd.DataFrame.from_dict(
            table, orient="index", columns=self.mdp.agent.actions)
        df.index = list(table.keys())

        goal_cell = self.grid.get_cell(goal_object.pos)
        goal_cell.type_ = "true_goal"

        self.draw_multiple(axs, df)
        fig.suptitle(
            "Policies for goal {}".format(
                goal_object.name),
            fontsize=16)

        plt.show()

    def trajectory(self, goal_object, trajectory):
        __, ax = plt.subplots(subplot_kw={"frame_on": False})
        grid = Grid.from_grid(self.grid, ax)
        grid.title = "Trajectory"

        goal_cell = grid.get_cell(goal_object)
        goal_cell.type_ = "true_goal"

        # trajectory path coloring
        for idx, pos in enumerate(trajectory):
            cell = grid.get_cell(pos)
            cell.type_ = "traj"
            cell.text += " " + str(idx)

        grid.draw()
        plt.show()


def plotPosteriors(posteriors, title="", labels=None):
    if not labels:
        labels = {}
    x = range(len(posteriors[:, 0]))
    plt.title(title)
    plt.plot(x, posteriors[:, 0])
    plt.plot(x, posteriors[:, 1])

    plt.legend(labels)
    plt.show()


def viewDictionaryStructureLevels(d, levels, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": " + str(key))
        if isinstance(value, dict):
            viewDictionaryStructureLevels(value, levels, indent + 1)
        else:
            print('\t' * (indent + 1) +
                  str(levels[indent + 1]) + ": " + str(value))


def viewDictionaryStructure(d, dictionaryType, indent=0):
    if dictionaryType == "t":
        levels = ["state", "action", "next state", "probability"]
    elif dictionaryType == "r":
        levels = ["state", "action", "next state", "reward"]
    elif dictionaryType == "t_key":
        levels = ["action", "next state", "probability"]
    elif dictionaryType == "r_key":
        levels = ["action", "next state", "reward"]
    viewDictionaryStructureLevels(d, levels, indent=indent)
