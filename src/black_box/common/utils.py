import os
import sys
from typing import Union, List

import numpy as np
from pandas import DataFrame, Series, melt, concat

from black_box.common.constants import META_DATASET_FOLDER_PATH


def check_array_elem(arr, idx, dflt):
    if arr is not None and idx < len(arr):
        return arr[idx]
    return dflt


def mapping(array: np.ndarray, keys, values):
    k = np.array(list(keys))
    v = np.array(list(values))

    mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)  # k,v from approach #1
    mapping_ar[k] = v
    return mapping_ar[array]


def add_param_to_dict(param_name: str, param_val, kwargs_dict: dict):
    kwargs_dict[param_name] = param_val
    return kwargs_dict


def get_rank_pos_from_score(score: DataFrame, higher_score_is_better: bool) -> DataFrame:
    # first in rank means position = 0
    rank_pos = DataFrame(index=score.index, columns=score.columns, dtype=np.int)
    index_to_sort = np.argsort(score.to_numpy(), axis=1)
    if higher_score_is_better:
        index_to_sort = index_to_sort[:, ::-1]
    for i, pos in zip(range(score.shape[1]), range(score.shape[1])):
        rank_pos.iloc[range(score.shape[0]), index_to_sort[:, i]] = pos
        # score.iloc[i, index_to_sort[i, :]] = [0, 1, 2]
    return rank_pos


def get_rank_from_score(score: DataFrame, higher_score_is_better: bool) -> DataFrame:
    rank = np.tile(score.columns, (score.shape[0], 1))
    index_to_sort = np.argsort(score.to_numpy(), axis=1)
    if higher_score_is_better:
        index_to_sort = index_to_sort[:, ::-1]
    return DataFrame(np.take_along_axis(rank, index_to_sort, axis=1), columns=range(score.shape[1]),
                     index=score.index, dtype=str)


def get_best_from_rank(rank: DataFrame):
    return rank.iloc[:, 0]


def get_best_from_score(score: DataFrame, higher_score_is_better: bool):
    return get_best_from_rank(get_rank_from_score(score, higher_score_is_better=higher_score_is_better))


def cut_third_dimension(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.reshape(y_pred, y_pred.shape[0:2])
    return y_pred


def sort_index_and_columns(df) -> Union[DataFrame, Series]:
    df = df.sort_index()
    if isinstance(df, DataFrame):
        df = df.reindex(columns=sorted(df.columns))
    return df


def lookup(df: DataFrame, columns: List):
    #columns_ = df.columns.get_indexer(columns)
    i = Series(columns, name="col", index=df.index)
    melted = melt(
        concat([i, df], axis=1),
        id_vars="col",
        value_vars=df.columns,
        ignore_index=False,
    )
    result = melted.loc[melted["col"] == melted["variable"], "value"]
    return result.loc[df.index]


def opes_factory(embed, rng, opes_type, model_name, output_type, working_dir):
    if opes_type == "regression":
        from black_box.estimator_selection.regression_estimator_selection_model import RegressionEstimatorSelectionModel
        opes = RegressionEstimatorSelectionModel(model_name=model_name, output_type=output_type, rng=rng,
                                                 single_y_for_train=True, embeddings=embed, working_dir=working_dir)
    else:
        print("ERROR: Off-Policy Estimator type 'opes_type' not known.")
        opes = None
    return opes


def get_metadataset_working_dir(rwd_type: str, n_points: int, n_bootstrap: int, avg: bool):
    return os.path.join(os.path.join(META_DATASET_FOLDER_PATH, get_metadataset_name(rwd_type, n_points, n_bootstrap),
                                     get_avg_folder_name(avg)))


def get_metadataset_name(reward_types: str, n_points: int, n_bootstrap: int):
    assert reward_types in ['bin', 'mix', 'cont']
    if n_bootstrap > 1:
        name_as_str_list = [reward_types, str(n_points) + "p", str(n_bootstrap) + "b"]
    else:
        name_as_str_list = [reward_types, str(n_points) + "p", "noboot"]
    suffix = "_".join(name_as_str_list)
    return suffix


def get_avg_folder_name(avg):
    return 'avg' if avg else 'noavg'
