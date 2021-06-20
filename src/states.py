from typing import Literal, NamedTuple, Callable, NoReturn, List

import numpy as np

from src.glb import SETTINGS
from src.data_tools.utils import debug


def bfs(states: List['State'],  # noqa: F821
        fun: Callable[['Graph', 'State'], NoReturn],  # noqa: F821
        graph=None):  # noqa: F821
    """ Do a breadth-first search on a graph structure.
    Parameters
    ----------
    states : [environment.State]
        List of all state objects in the environment
    fun : function [Graph, State] -> no return
    graph : Graph, optional
        only needed if fun needs graph instance
    """
    i = 0
    visited, queue = set(), states[0:1]
    while queue:
        state = queue.pop(0)
        if state not in visited:
            #print(i / len(states))
            i += 1

            fun(graph, state)
            visited.add(state)

            next_states = state.reachable_states()
            queue.extend(next_states - visited)
    return visited


@debug
def setup_states(states, container):
    """ Generate the links between states
    """
    #print("num states:", len(states))

    def fun(__, state_):
        for action in SETTINGS.possible_actions:
            next_state = container.get_state(state_ + action)
            if not next_state:
                # if action not possible, stay in state
                next_state = state_
            state_.next[action, next_state] = 1

    for state in states:
        fun(None, state)


class Transition(NamedTuple):
    state: 'State'
    action: int  # : tuple[int, int] # TODO 3.9
    next_state: 'State'
    value: float

    def __eq__(self, other):
        return (self.state == other.state) and\
               (self.action == other.action) and\
               (self.next_state == other.next_state) and\
               (self.value == other.value)

    def __str__(self):
        return f"from ({self.state.get_coords()}) to \
            ({self.next_state.get_coords()}) by {self.action}: {self.value}"

    def __repr__(self):
        return self.__str__()


class TransitionTable:
    def __init__(self, state):
        self.state = state
        self._transitions = []

    def __iter__(self):
        return iter(self._transitions)

    def __next__(self):
        return next(self._transitions)

    def items(self):
        return [((t.action, t.next_state), t.value) for t in self.__iter__()]

    def keys(self):
        return [(t.action, t.next_state) for t in self.__iter__()]

    def __setitem__(self, key, value):
        action, next_state = key
        transition = self.__getitem__(key)
        if self.__getitem__(key):  # exists
            transition.prob = value
        else:
            # not existent
            self._transitions.append(Transition(
                self.state, action, next_state, value))

    def __getitem__(self, key):
        if isinstance(key[0], tuple):
            # action, next_state
            return list(
                filter(
                    lambda trans: trans.action == key[0] and trans.next_state == key[1],
                    self._transitions))
        # action
        return list(filter(lambda trans: trans.action ==
                           key, self._transitions))

    def __str__(self):
        s = f"Transitions from {self.state}: \n"
        for t in self._transitions:
            s += str(t)
        return s

    def __repr__(self):
        return self.__str__()


class State:
    def __init__(self, pos):
        self.next = TransitionTable(self)
        self.__pos = pos

    def reachable_states(self):
        return {trans.next_state for trans in self.next}

    def get_coords(self):
        return self.__pos

    @property
    def actions(self):
        return sorted(list({t.action for t in self.next}))

    def distance(self, goal):
        pass

    def __eq__(self, other):
        if isinstance(other, State):
            return self.get_coords() == other.get_coords()
        return self.get_coords() == other


class GPSState(tuple, State):
    def __new__(cls, tup, *args):
        return super(GPSState, cls).__new__(cls, tup)

    def __init__(self, tup, *args):
        super(GPSState, self).__init__(tup)
        State.__init__(self, tup)

    def distance(self, goal):
        return DISTANCE[SETTINGS.norm](self - goal.pos)

    def __add__(self, other):
        assert len(self) == len(other)
        return tuple.__new__(GPSState, (self[i] + other[i]
                                        for i in range(len(self))))

    def __sub__(self, other):
        assert len(self) == len(other)
        return tuple.__new__(GPSState, (self[i] - other[i]
                                        for i in range(len(self))))


def unit_vector(v):
    if np.linalg.norm(v) == 0:
        return (0, 0)
    return v / np.linalg.norm(v)


DISTANCE = {"L1": lambda action: np.linalg.norm(action, ord=1),
            "L2": lambda action: np.linalg.norm(action)}


class DistanceDirectionState(State):
    # TODO Orientation
    # TODO Direction 'neurons' with fire power

    def __init__(self, pos, goals):
        super(DistanceDirectionState, self).__init__(pos)

        self.goal_names = [goal.name for goal in goals]
        self.vec_to_goals = {
            goal.name: [
                goal.pos[i] -
                pos[i] for i in range(
                    len(pos))] for goal in goals}

    def distance(self, goal):
        return DISTANCE[SETTINGS.norm](self.vec_to_goals[goal.name])

    def direction(self, goal):
        return unit_vector(self.vec_to_goals[goal])

    def __add__(self, action):
        if isinstance(action, DistanceDirectionState):
            raise ValueError("should be an action, not a state")

        new_ = DistanceDirectionState(
            tuple(
                self.get_coords()[i] +
                action[i] for i in range(
                    len(action))),
            [])

        new_.goal_names = self.goal_names
        new_.vec_to_goals = {g: [
            v[i] - action[i] for i in range(len(v))] for g, v in self.vec_to_goals.items()}

        return new_

    def __key(self):
        vecs = [(k, tuple(v)) for k, v in self.vec_to_goals.items()]
        vecs.append(self.get_coords())
        return tuple(vecs)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return f"dist_dir at (pos: {self.get_coords()})"

    def __repr__(self):
        return self.__str__()
