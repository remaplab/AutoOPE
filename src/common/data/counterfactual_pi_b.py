# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import copy

import numpy as np


def get_counterfactual_action_distribution(dataset, cf_beta, n_rounds):
    """
    Get action distribution for counterfactual beta.
    Note that we need to use this function before we get factual batch data by dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    Args:
        dataset (obp.dataset.SyntheticBanditDataset): original synthetic data generator
        cf_beta (float): counterfactual beta
        n_rounds (int): sample size

    Returns:
        np.array: action distribution for counterfactual beta
    """
    cf_dataset = copy.deepcopy(dataset)
    setattr(cf_dataset, 'beta', cf_beta)
    return cf_dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)['pi_b']


def get_counterfactual_bandit_feedback(dataset, cf_beta, cf_policy_function, n_rounds, cf_n_deficient_actions: int = 0):
    """
    Get action distribution for counterfactual beta.
    Note that we need to use this function before we get factual batch data by dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    Args:
        dataset: original synthetic data generator
        cf_beta (float): counterfactual beta
        n_rounds (int): sample size
        cf_policy_function
        cf_n_deficient_actions

    Returns:
        np.array: action distribution for counterfactual beta
    """
    cf_dataset = copy.deepcopy(dataset)
    setattr(cf_dataset, 'beta', cf_beta)
    setattr(cf_dataset, 'behavior_policy_function', cf_policy_function)
    setattr(cf_dataset, 'cf_n_deficient_actions', cf_n_deficient_actions)
    return cf_dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)


def get_counterfactual_pscore(pi_e, log_actions, log_positions):
    if len(pi_e.shape) == 3 and pi_e.shape[2] > 1:
        pscore = pi_e[np.arange(pi_e.shape[0]), log_actions, log_positions]
    else:
        pscore = pi_e[np.arange(pi_e.shape[0]), log_actions, :].flatten()
    return pscore
