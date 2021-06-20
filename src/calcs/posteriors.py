"""
Module to compute the posteriors P(trajectory | Goal)
p(s0, s1 | Goal=goal) = p(s0 | Goal=goal) * policy(s0->s1)
"""
from pprint import pprint
import numpy as np

#from src.data_tools.utils import debug


def probability_next_state(state, next_, policy):
    """ Probability under policy pi(state->next_)
    Parameters
    ----------
    state : graph.State
            the current state in the trajectory.
    next_ : graph.State
            the next state in the trajectory.
    policy: dict {state:{action:reward}}
            Boltzmann policies.
    """
    possible_transitions_to_next = list(
        filter(lambda t: t.next_state.get_coords() == next_, state.next))
    if not possible_transitions_to_next:
        raise ValueError("states not connected: {} {}.\
            If the states are the same, you probably need to enable (0,0) as a possible action in the settings.".format(state, next_))
    return sum(
        t.value * policy[state][t.action] for t in possible_transitions_to_next
    )


def _sequence_of_state_probabilities(policies, priors, trajectory):
    """
    Arguments
    ---------
    policies: list(dict)
        List of length #goals, dicts of policy for each goal
    priors: np.array
        Priors of length #goals
    trajectory: list(State)
        list of states in trajectory (state objects, from respective model)

    Returns
    -------
    np.array len(policies) x len(trajectory)
    """
    probNextState = [np.log(priors)]
    for t, state in enumerate(trajectory[:-1]):
        nextState = trajectory[t+1]
        probNextState.append([np.log(probability_next_state(state, nextState, policy))
                              for policy in policies])
    return np.cumsum(np.array(probNextState), axis=0)


# Posteriors
# @debug
def get_posteriors(policies, priors, trajectory):
    """
    p(trajectory | goal) * priors
    p(traj | goal) = prod(policy(s(t), a | goal)) for t in traj

    Parameters
    ----------
    policies : {goal:{state : {action : probability}}}
        Boltzmann policies for all goals.
    priors : [probability]
        Priors for all goals. Sums up to 1.
    trajectory : [State]
    """
    policies = list(policies.values())

    pos_neg_log = _sequence_of_state_probabilities(
        policies,
        priors,
        trajectory)

    pos_neg_log = np.exp(pos_neg_log)
    pos_neg_log *= np.array(priors)

    # normalize every row
    sums = pos_neg_log.sum(axis=1, keepdims=True)

    return np.divide(
        pos_neg_log,
        sums,
        out=np.zeros_like(pos_neg_log),
        where=sums != 0)  # posterior / posterior.sum(axis=1, keepdims=True)


def model_posteriors(trajectory, true_goal, models, model_priors=None):
    if not model_priors:
        model_priors = np.full(len(models), 1 / len(models))

    posteriors = np.zeros((len(trajectory), len(models)))
    for idx, (model, prior) in enumerate(zip(models, model_priors)):
        policy = [model.get_policies()[true_goal]]
        _trajectory = model.mdp.trajectory_to_agent_states(trajectory)
        seq_prob = _sequence_of_state_probabilities(
                    policy,
                    [prior],
                    _trajectory)
        posteriors[:, idx] = seq_prob.T

    pos_neg_log = np.exp(posteriors)
    pos_neg_log *= np.array(model_priors)

    sums = pos_neg_log.sum(axis=1, keepdims=True)

    return np.divide(
        pos_neg_log,
        sums,
        out=np.zeros_like(pos_neg_log),
        where=sums != 0)  # posterior / posterior.sum(axis=1, keepdims=True)
