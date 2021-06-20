import math
from pprint import pprint
import numpy as np

#from src.data_tools.utils import debug

# TODO MDP and MDPSolver
# TODO move beta to class


class Solver:
    def __init__(self, mdp, beta=.4, gamma=.95):
        self.mdp = mdp
        self.beta = beta
        self.gamma = gamma

    def policies(self, goal, beta):
        # TODO calculate_policies
        raise NotImplementedError


class GreedySolver(Solver):
    def __init__(self, mdp, gamma=.95):
        super().__init__(mdp, gamma=gamma)
        self.utilities = {}

    def _q_value(self, state, action, goal):
        """
        """
        return sum(
            transition.value * self.reward_table[transition]
            + self.gamma * self.utilities[goal][transition.next_state]
            for transition in state.next[action]
        )

    def value_iteration(self, goal, convergence_tolerance=10e-7):  # e-7
        self.utilities[goal] = {state: 0 for state in self.mdp.agent_states}

        for state, utility in self.utilities[goal].items():
            # utility is q value of the best possible next state
            self.utilities[goal][state] = max(self._q_value(
                state, action, goal) for action in self.mdp.agent.actions)

        return self.utilities[goal]

    # @debug
    def policies_(self, goal, beta):
        """ Boltzmann policies / soft-max functions of Q
        P_policy(a_t | s_t, goal) proportional to exp(beta*Q_policy_g(s_t, a_t))
        """
        self.reward_table = self.mdp.reward_table(goal)
        self.value_iteration(goal)
        return {state: self.get_boltzmann_policy(
            state, goal, beta) for state in self.mdp.agent_states}

    def policies(self, goal, beta):
        """
        Returns
        -------
        dict goal: state: action: probability
        """
        # ditance / sum(distance)
        self.reward_table = {
            t: self.mdp.R(
                t.state,
                t.action,
                goal) for t in self.mdp.T}
        self.reward_table.update(
            {t: goal.reward + self.reward_table[t] for t in self.mdp.T if t.next_state.get_coords() == goal.pos})

        max_distance = max(s.distance(goal)
                           for s in self.mdp.agent_states)

        self.utilities[goal] = {state: (
            max_distance - state.distance(goal)) for state in self.mdp.agent_states}

        return {state: self.get_boltzmann_policy(
            state, goal, beta) for state in self.mdp.agent_states}

    def get_boltzmann_policy(self, state, goal, beta, print_statements=False):
        """
        Parameters
            ----------
            state : State
            beta : [0,1], optional
                noise factor. default is .4. when beta=0, we have random walk.
        """
        exponents = [beta * self._q_value(state, action, goal)
                     for action in state.actions]
        # Scale to [0,700] if there are exponents larger than 700
        if any(abs(exponent) > 700 for exponent in exponents):
            if print_statements:
                print("scaling exponents to [0,700]... On State:", "\n", state)
            exponents = [np.sign(exponent) * 700 * (exponent / max(exponents))
                         for exponent in exponents]

        state_policy = {action: math.exp(exponent)
                        for exponent, action in zip(exponents, state.actions)}
        # normalize and return
        return {key: val / sum(state_policy.values())
                for key, val in state_policy.items()}


class OptimalSolver(Solver):
    """
    Parameters
    ----------
    mdp : src.model.model.MDP
    """

    def __init__(self, mdp, gamma=.95):
        super().__init__(mdp, gamma=gamma)

        self.utilities = {}
        self.gamma = gamma
        self.reward_table = {}  # TODO make into cached versions, this is terrible

    def _q_value(self, state, action, goal):
        """
        """
        return sum(
            transition.value * self.reward_table[transition]
            + self.gamma * self.utilities[goal][transition.next_state]
            for transition in state.next[action]
        )

    # @debug
    # TODO make private
    def value_iteration(self, goal, convergence_tolerance=10e-7):
        delta = convergence_tolerance * 100
        self.utilities[goal] = {state: 0 for state in self.mdp.agent_states}
        while delta > convergence_tolerance:
            delta = 0
            for state, utility in self.utilities[goal].items():
                # utility is q value of the best possible next state
                self.utilities[goal][state] = max(self._q_value(
                    state, action, goal) for action in self.mdp.agent.actions)

                delta = max(delta, abs(utility - self.utilities[goal][state]))

        return self.utilities[goal]

    # @debug
    def policies(self, goal, beta):
        """ Boltzmann policies / soft-max functions of Q
        P_policy(a_t | s_t, goal) proportional to exp(beta*Q_policy_g(s_t, a_t))
        """
        self.reward_table = self.mdp.reward_table(goal)
        self.value_iteration(goal)

        return {state: self.get_boltzmann_policy(
            state, goal, beta) for state in self.mdp.agent_states}

    # TODO make private
    def get_boltzmann_policy(self, state, goal, beta, print_statements=False):
        """
        Parameters
            ----------
            state : State
            beta : [0,1], optional
                noise factor. default is .4. when beta=0, we have random walk.
        """
        exponents = [beta * self._q_value(state, action, goal)
                     for action in state.actions]

        # Scale to [0,700] if there are exponents larger than 700
        if any(abs(exponent) > 700 for exponent in exponents):
            if print_statements:
                print("scaling exponents to [0,700]... On State:", "\n", state)
            exponents = [np.sign(exponent) * 700 * (exponent / max(exponents))
                         for exponent in exponents]

        state_policy = {action: math.exp(exponent)
                        for exponent, action in zip(exponents, state.actions)}

        # normalize and return
        return {key: val / sum(state_policy.values())
                for key, val in state_policy.items()}
