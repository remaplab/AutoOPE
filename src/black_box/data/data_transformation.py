import enum

import numpy as np
import pandas as pd

from black_box.common.utils import get_rank_pos_from_score


class OutputType(enum.Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, high_score_better: bool):
        self.high_score_better = high_score_better

    NO_TRANSFORMATION = False
    TOP1 = True
    RANK_RWD = True
    ERROR_RWD = True


def transform_output(estim_errors: pd.DataFrame, estim_errors_transformation: OutputType) -> pd.DataFrame:
    if estim_errors_transformation.name == OutputType.NO_TRANSFORMATION.name:
        pass
    elif estim_errors_transformation.name == OutputType.RANK_RWD.name:
        estim_errors = _get_rank_as_rwd(estim_errors)
    elif estim_errors_transformation.name == OutputType.TOP1.name:
        estim_errors = _get_top1(estim_errors)
    elif estim_errors_transformation.name == OutputType.ERROR_RWD.name:
        estim_errors = _get_error_as_rwd(estim_errors)
    else:
        print('Error: trasformation', estim_errors_transformation.value, 'not available')
        estim_errors = None
    return estim_errors


def _get_rank_as_rwd(errors: pd.DataFrame):
    rank_pos = get_rank_pos_from_score(errors, higher_score_is_better=False)
    rank_pos_arr = rank_pos.to_numpy()
    rewards = - rank_pos_arr + np.max(rank_pos_arr)
    return pd.DataFrame(rewards, columns=rank_pos.columns, index=rank_pos.index, dtype=int)


def _get_top1(errors: pd.DataFrame):
    rank_pos = get_rank_pos_from_score(errors, higher_score_is_better=False)
    rank_pos_arr = rank_pos.to_numpy()
    rewards = np.zeros(rank_pos_arr.shape)
    mask = rank_pos_arr == np.min(rank_pos_arr)
    rewards[mask] = np.ones(rank_pos_arr.shape)[mask]
    return pd.DataFrame(rewards, columns=rank_pos.columns, index=rank_pos.index, dtype=int)


def _get_error_as_rwd(op_squared_errors_df: pd.DataFrame):
    rwd = (- op_squared_errors_df).add(op_squared_errors_df.max(axis=1), axis=0)
    return rwd
