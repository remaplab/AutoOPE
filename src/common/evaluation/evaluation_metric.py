# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def calculate_relative_regret_e(true_data, estimated_data, estimator_selection_metrics='mse', random=None):
    """calculate relative regret for evaluation of estimator selection

    Args:
        true_data (pd.DataFrame): dataframe including estimator name, estimator_selection_metric, and rank
        estimated_data (pd.DataFrame): dataframe including estimator name and rank
        estimator_selection_metrics (str, optional): Defaults to 'mse'. Must be 'mse' or 'mean relative-ee'
        random: random state used to break ties

    Returns:
        float: relative regret
    """

    predicted_best_estimator_name_list = estimated_data['estimator_name'][estimated_data['rank'] == 1].values
    predicted_best_estimator_name = predicted_best_estimator_name_list[0]
    random_est = None
    if len(predicted_best_estimator_name_list) > 1:
        predicted_best_estimator_name = random.choice(predicted_best_estimator_name_list)
        random_est = predicted_best_estimator_name

    true_best_estimator_name = true_data['estimator_name'][true_data['rank'] == 1].values[0]

    predicted_estimator_performance = true_data[estimator_selection_metrics][true_data['estimator_name']
                                                                             == predicted_best_estimator_name].values[0]
    true_estimator_performance = true_data[estimator_selection_metrics][true_data['estimator_name']
                                                                        == true_best_estimator_name].values[0]

    relative_regret_e = (predicted_estimator_performance / true_estimator_performance) - 1.0

    return relative_regret_e, random_est



def calculate_mse_e(true_data, estimated_data, estimator_selection_metrics='mse', random_est=None):
    predicted_best_estimator_name_list = estimated_data['estimator_name'][estimated_data['rank'] == 1].values
    predicted_best_estimator_name = predicted_best_estimator_name_list[0]
    if len(predicted_best_estimator_name_list) > 1 and random_est is not None:
        predicted_best_estimator_name = random_est

    predicted_estimator_performance = true_data[estimator_selection_metrics][true_data['estimator_name']
                                                                             == predicted_best_estimator_name].values[0]
    return predicted_estimator_performance


def calculate_rank_correlation_coefficient_e(true_data, estimated_data):
    """calculate rank correlation coefficient for evaluation of estimator selection

    Args:
        true_data (pd.DataFrame): dataframe including estimator name and rank
        estimated_data (pd.DataFrame): dataframe including estimator name and rank

    Returns:
        float: rank correlation coefficient
    """

    merged_data = pd.merge(
        true_data.rename(columns={'rank': 'rank_true'}),
        estimated_data.rename(columns={'rank': 'rank_predicted'}),
        how='left', 
        on='estimator_name'
        )

    rank_true = merged_data['rank_true'].values
    rank_predict = merged_data['rank_predicted'].values
    try:
        rank_cc, pvalue = spearmanr(rank_predict, rank_true, nan_policy='omit')
    except ValueError as e:
        print(e, "Setting rank correlation coefficient to 0.0")
        rank_cc, pvalue = 0.0, 0.0

    # if rank_predict is constant (e.g. in Random Estimator Selection method the rank is 1 for all estimators)
    if rank_cc is np.nan and pvalue is np.nan:
        rank_cc = 0.0

    return rank_cc


def calculate_relative_regret_p(true_data, estimated_data, random=None):
    """calculate relative regret for evaluation of policy selection

    Args:
        true_data (pd.DataFrame): dataframe including policy name, policy value, and rank
        estimated_data (pd.DataFrame): dataframe including policy name and rank
        random: random state used to break ties

    Returns:
        float: relative regret
    """

    predicted_best_policy_name_list = estimated_data['policy_name'][estimated_data['rank'] == 1].values
    predicted_best_policy_name = predicted_best_policy_name_list[0]
    if len(predicted_best_policy_name_list) > 1:
        predicted_best_policy_name = random.choice(predicted_best_policy_name_list)
    true_best_policy_name = true_data['policy_name'][true_data['rank'] == 1].values[0]

    predicted_policy_performance = true_data['policy_value'][true_data['policy_name'] ==
                                                             predicted_best_policy_name].values[0]
    true_policy_performance = true_data['policy_value'][true_data['policy_name'] == true_best_policy_name].values[0]

    relative_regret_p = (predicted_policy_performance / true_policy_performance) - 1.0
    relative_regret_p = (-1.0) * relative_regret_p

    return relative_regret_p


def calculate_rank_correlation_coefficient_p(true_data, estimated_data):
    """calculate rank correlation coefficient for evaluation of policy selection

    Args:
        true_data (pd.DataFrame): dataframe including policy name and rank
        estimated_data (pd.DataFrame): dataframe including policy name and rank

    Returns:
        float: rank correlation coefficient
    """

    merged_data = pd.merge(
        true_data.rename(columns={'rank': 'rank_true'}),
        estimated_data.rename(columns={'rank': 'rank_predicted'}),
        how='left', 
        on='policy_name'
        )

    rank_true = merged_data['rank_true'].values
    rank_predict = merged_data['rank_predicted'].values
    rank_cc, pvalue = spearmanr(rank_predict, rank_true)

    return rank_cc
