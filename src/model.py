from src.settings import Settings
from typing import NamedTuple, Literal
import itertools as it
import math

import numpy as np
import pandas as pd

from src import states
from src.calcs import mdp
from src.glb import SETTINGS
from src.data_tools import io
from src.calcs import posteriors


DISTANCE = {"L1": lambda action: np.linalg.norm(action, ord=1),
            "L2": lambda action: np.linalg.norm(action)}


COST_OBSTACLE = -100.0


class Model:
    def __init__(self,
                 env,
                 solver: Literal["Optimal",
                                 "Greedy"] = "Optimal",
                 state_type=SETTINGS.state_type,
                 determinism=SETTINGS.determinism):
        self.env = env
        self.agent = Agent(
            representation_type=state_type,
            determinism=determinism)
        self.mdp = MDP(self.env, self.agent, state_type=state_type)
        self.solver = mdp.__dict__["{}Solver".format(solver)](
            self.mdp)

        self.__policies = None

    @property
    def goals(self):
        return self.env.goals

    @property
    def obstacles(self):
        return self.env.obstacles

    def get_policies(self, goal_rewards=None) -> dict:
        if self.__policies:
            return self.__policies
        if goal_rewards:
            raise NotImplementedError
        self.__policies = {goal: self.solver.policies(
            goal, beta=self.agent.beta) for goal in self.env.goals}
        return self.__policies

    def get_posteriors(
            self,
            policies,
            trajectory,
            goal_priors=None) -> np.array:
        trajectory = self.mdp.trajectory_to_agent_states(trajectory)
        if not goal_priors:
            goal_priors = np.full(len(self.env.goals), 1 / len(self.env.goals))
        return posteriors.get_posteriors(
            policies, goal_priors, trajectory)


class SamplingModel(Model):
    def __init__(
            self,
            dimensions,
            goals,
            obstacles,
            start_pos,
            solver="Optimal",
            goal_priors=None,
            state_type=SETTINGS.state_type,
            determinism=SETTINGS.determinism):
        super().__init__(
            Environment(
                dimensions,
                goals,
                obstacles),
            solver=solver,
            state_type=state_type,
            determinism=determinism)
        self.solver_type = solver
        self.agent.start = self.mdp.get_state(start_pos)
        self.goal_priors = goal_priors
        self.dims = dimensions

    @staticmethod
    def from_file(
            file_name,
            pixelation=1,
            state_type=SETTINGS.state_type,
            determinism=SETTINGS.determinism) -> "SamplingModel":
        # TODO get these things from data:
        dimensions = None
        goals = None
        obstacles = None
        start_pos = None
        raise NotImplementedError
        return SamplingModel(
            dimensions,
            goals,
            obstacles,
            start_pos,
            state_type=state_type,
            determinism=determinism)

    def sample_goal(self) -> "Goal":
        np.random.seed(SETTINGS.seed)
        goal_idx = np.random.choice(
            len(self.env.goals), p=self.goal_priors)
        return self.env.goals[goal_idx]

    def sample_trajectory(self, goal: "Goal", policy):
        position = self.agent.start
        trajectory = [position]

        # allow max. 3000 steps
        while position != goal.pos and len(trajectory) < 3000:

            # take action probabilistically
            s_ = self.mdp.get_state(position)

            prob_action = [policy[s_][action] for action in s_.actions]
            np.random.seed(SETTINGS.seed)
            action_idx = np.random.choice(
                len(s_.actions), size=1, p=prob_action)
            sampled_action = s_.actions[int(action_idx)]

            # get new position
            sampled_idx = np.random.choice(len(s_.next[sampled_action]), size=1, p=[
                                           t.value for t in s_.next[sampled_action]])[0]
            new_position = s_.next[sampled_action][sampled_idx].next_state

            # update to new position and add to trajectory
            position = new_position
            trajectory.append(position)
        return trajectory

    def sample(self, goal=None) -> list:

        if self.solver_type == "Greedy" and self.mdp.agent.beta > 1:
            print("WARNING: if obstacles are concave, might take forever")

        if not goal:
            # Sample a goal
            # uniform random sample of goal
            goal_idx = np.random.choice(
                len(self.env.goals), p=self.goal_priors)
            goal = self.env.goals[goal_idx]

        # Sample a trajectory under that goal
        trajectory = self.sample_trajectory(goal, self.get_policies()[goal])
        #trajectory = self.mdp.trajectory_to_agent_states(trajectory)

        return trajectory


class DataModel(Model):
    def __init__(
            self,
            file_name,
            goals,
            obstacles,
            pixelation=1,
            solver="Optimal",
            state_type=SETTINGS.state_type,
            determinism=SETTINGS.determinism):
        self.reader = io.CSVReader(file_name, pixelation=pixelation)
        self.df = None
        env = self.load(file_name, goals, obstacles, pixelation)
        super().__init__(
            env,
            solver=solver,
            state_type=state_type,
            determinism=determinism)

    def load(
            self,
            file_name,
            goals,
            obstacles,
            pixelation=1,
            test_phase=False) -> "Environment":
        self.reader.pixelation = pixelation

        if test_phase:
            self.df = self.reader.test_phase(file_name)
        else:
            self.df = self.reader._single_file(file_name)

        # apply pixelation to goals and obstacles
        transform = np.array(io.get_center_transform(file_name=file_name))
        transform /= pixelation

        goals = [Goal(g.name, tuple(math.floor(coord / pixelation - center_coord) for coord, center_coord in zip(g.pos, transform))) for g in goals]
        obstacles = [tuple(math.floor(coord / pixelation - center_coord) for coord, center_coord in zip(t, transform)) for t in obstacles]

        return Environment(self.reader.dimensions(file_name), goals, obstacles)

    def trajectory(self):
        return self.reader.trajectory(self.df)
    
    @property
    def goals(self):
        return self.env.goals
    
    @property
    def obstacles(self):
        return self.env.obstacles


class Agent:
    def __init__(
            self,
            start_pos=None,
            representation_type=SETTINGS.state_type,
            determinism: float = SETTINGS.determinism):
        self.start = start_pos
        self.actions = SETTINGS.possible_actions
        self.state_type = representation_type
        self.beta = determinism

    def move_cost(self, action: tuple):
        # TODO orientation change is costly :)
        if action == (0, 0):
            return -1.0
        return -DISTANCE[SETTINGS.norm](action)


class Goal(NamedTuple):
    name: str   # name of the goal
    pos: int  # tuple[int, int]  # position in environment # TODO 3.9
    reward: float = 10.0


class Environment:
    def __init__(self, dims, goals, obstacles):
        self.goals = goals
        self.obstacles = obstacles

        self.df = pd.DataFrame(
            0.0, columns=list(
                io.Factory(
                    **dims).build()), index=["reward"])

        try:
            [self[goal.pos] for goal in goals]
            [self[obstacle] for obstacle in obstacles]
        except KeyError as exc:
            raise ValueError(f"goal or obstacle at {exc.args[0]} is not in the environment.")

        for obstacle in obstacles:
            self[obstacle] = COST_OBSTACLE

        for goal in self.goals:
            self[goal.pos] = goal.name

    @property
    def coords(self):
        return self.df.columns

    def get_goal(self, key):
        if isinstance(key, str):
            return [g for g in self.goals if g.name == key][0]
        return [g for g in self.goals if g.pos == key][0]

    def __getitem__(self, key):
        return self.df[key]["reward"]

    def __setitem__(self, key, value):
        self.df.loc["reward", [key]] = value


class MDP:
    def __init__(
            self,
            environment: Environment,
            agent: Agent,
            state_type=SETTINGS.state_type):
        self.env = environment
        self.agent = agent

        if self.agent.start and self.agent.start not in self.env.coords:
            raise ValueError("agent starting position not in environment")

        self.agent_states = [states.__dict__[state_type](
            state, self.env.goals) for state in (set(self.env.coords) - set(self.env.obstacles))]
        states.setup_states(self.agent_states, self)

    @property
    def T(self):
        """ All possible transitions in the environment
        """
        return list(it.chain.from_iterable(
            [iter(state.next) for state in self.agent_states]))

    def trajectory_to_agent_states(self, trajectory):
        """ Turns a list of tuples into State objects in graph """
        return list(map(self.get_state, trajectory))

    def reward_table(self, goal):
        return {t: self.R(t.state, t.action, goal) for t in self.T}

    def R(self, state, action, goal):
        """ Reward function.
        """
        reward = self.env[state.get_coords()]
        if isinstance(reward, str):
            reward = goal.reward if reward == goal.name else 0.0
        if reward == COST_OBSTACLE:
            return float(reward)

        return reward + self.agent.move_cost(action)

    def get_state(self, state):
        return next((s_ for s_ in self.agent_states if s_ == state), None)
