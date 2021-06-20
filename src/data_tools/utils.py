import pandas as pd
from time import time

import logging


logger = logging.getLogger()


def debug(fn):

    def wrapper(*args, **kwargs):
        logger.debug("Entering {:s}...".format(fn.__name__))
        ts = time()
        result = fn(*args, **kwargs)
        te = time()
        logger.debug("Finished {:s}.".format(fn.__name__))
        logger.debug('func:%r took: %2.4f sec' %
                     (fn.__name__, te - ts))
        return result

    return wrapper


def validate_inputs(goals, models, start_pos: tuple = None):
    for model in models:
        # goals
        if any([goal.pos not in model.env.coords for goal in goals]):
            raise ValueError("at least one goal is not in the environment")
        # agent starting position
        if start_pos and start_pos not in model.env.coords:
            raise ValueError("starting position not in environment")


def dict_to_dataframe(dict_):
    """ Turn a dict of transition : value to a pandas dataframe.

    Parameters
    ----------
    dict_ : {states.Transition : float}
        policy table

    Returns
    -------
    pandas.DataFrame
        states as rows/index, actions as columns, dict values as values in df
    """

    actions = sorted(list(set((key.action for key in list(dict_.keys())))))
    new_dict = {action: [] for action in actions}

    for transition, val in dict_.items():
        new_dict[transition.action].append(val)

    states = list(dict.fromkeys([k.state.get_coords()
                                 for k in dict_.keys()]))  # unique states

    df = pd.DataFrame.from_dict(new_dict)
    df.columns = actions
    df.index = states

    return df
