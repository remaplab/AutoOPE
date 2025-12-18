import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils import check_consistent_length

from black_box.common.utils import lookup


def mse(y: pd.DataFrame, y_pred: pd.DataFrame):
    y_type, y, y_pred, multioutput = _check_reg_targets(y, y_pred, None)
    check_consistent_length(y, y_pred, None)
    output_errors = (y - y_pred) ** 2
    return np.average(output_errors)


def rmse(y: pd.DataFrame, y_pred: pd.DataFrame):
    mse_ = mse(y, y_pred)
    return np.sqrt(mse_)


def mae_perc(y: pd.DataFrame, y_pred: pd.DataFrame):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y, y_pred, None)
    check_consistent_length(y_true, y_pred, None)
    epsilon = np.finfo(np.float64).eps
    output_errors = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(output_errors)


def mae(y: pd.DataFrame, y_pred: pd.DataFrame):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y, y_pred, None)
    check_consistent_length(y_true, y_pred, None)
    output_errors = np.abs(y_true - y_pred)
    return np.average(output_errors)


def kt_corr(y: pd.DataFrame, y_pred: pd.DataFrame):
    raw_values = [kendalltau(y.iloc[i], y_pred.iloc[i])[0] for i in range(y_pred.shape[0])]
    return np.average(np.array(raw_values))


def acc(y: pd.DataFrame, y_pred: pd.DataFrame):
    acc_raw_val = (y == y_pred)
    return np.average(acc_raw_val)


def error_relative_regret(y: pd.DataFrame, y_pred: pd.DataFrame):
    return relative_regret(y=y, y_pred=y_pred, high_score_better=False)


def error_regret(y: pd.DataFrame, y_pred: pd.DataFrame):
    return regret(y=y, y_pred=y_pred, high_score_better=False)


def relative_regret(y: pd.DataFrame, y_pred: pd.DataFrame, high_score_better: bool = False):
    best_pred_estimator_cols = y_pred.idxmax(axis=1) if high_score_better else y_pred.idxmin(axis=1)
    best_pred_estimator_performance = lookup(y, best_pred_estimator_cols.to_list())
    best_estimator_performance = np.max(y, axis=1) if high_score_better else np.min(y, axis=1)

    rel_regret = (best_pred_estimator_performance / best_estimator_performance) - 1.0
    rel_regret = np.abs(rel_regret)

    return np.average(rel_regret)


def regret(y: pd.DataFrame, y_pred: pd.DataFrame, high_score_better: bool = False):
    best_pred_estimator_cols = y_pred.idxmax(axis=1) if high_score_better else y_pred.idxmin(axis=1)
    best_pred_estimator_performance = lookup(y, best_pred_estimator_cols.to_list())
    best_estimator_performance = np.max(y, axis=1) if high_score_better else np.min(y, axis=1)

    regret_val = best_pred_estimator_performance - best_estimator_performance
    regret_val = np.abs(regret_val)

    return np.average(regret_val)


def spearman_corr(y: pd.DataFrame, y_pred: pd.DataFrame):
    spearman_res = [spearmanr(y.iloc[i], y_pred.iloc[i])[0] for i in range(y_pred.shape[0])]
    return np.average(np.array(spearman_res))


def metric_plot_label(metric: str):
    if metric == "MAE_PERC":
        return "MAE Percentage"
    elif metric == "REL_REGRET":
        return "Relative Regret"
    elif metric == "REGRET":
        return "Regret"
    elif metric == "KENDALL_TAU":
        return "Kendall's Tau Correlation"
    elif metric == "SPEARMAN":
        return "Spearman's Rank Correlation"
    elif metric == "TOP1_ACC":
        return "Top-1 Accuracy"
    else:
        return metric


ERROR_METRICS = ["MSE", "RMSE", "MAE", "MAE_PERC"]
ERROR_METRICS_FUN_DICT = {"MSE": mse,
                          "RMSE": rmse,
                          "MAE": mae,
                          "MAE_PERC": mae_perc}

REGRET_METRICS = ["REL_REGRET", "REGRET"]
REGRET_METRICS_FUN_DICT = {"REL_REGRET": error_relative_regret,
                           "REGRET": error_regret}

RANKING_METRICS = ["KENDALL_TAU", "SPEARMAN"]
RANKING_METRICS_FUN_DICT = {"KENDALL_TAU": kt_corr,
                            "SPEARMAN": spearman_corr}

CLASSIFICATION_METRICS = ["TOP1_ACC"]
CLASSIFICATION_METRICS_FUN_DICT = {"TOP1_ACC": acc}

ALL_METRICS = ERROR_METRICS + CLASSIFICATION_METRICS + RANKING_METRICS + REGRET_METRICS
ALL_METRICS_FUN_DICT = ERROR_METRICS_FUN_DICT | CLASSIFICATION_METRICS_FUN_DICT | RANKING_METRICS_FUN_DICT | REGRET_METRICS_FUN_DICT
