import random
from copy import deepcopy

import numpy as np
import pandas as pd
from obp.ope.helper import estimate_student_t_lower_bound
from sklearn.utils import check_random_state

from common.estimator_selection.base_estimator_selection import BaseEstimatorSelection



class SLOPEEstimatorSelection(BaseEstimatorSelection):
    """
    Automatic off-policy estimator selection method for OPE
    """

    def __init__(self, ope_estimators, q_models, estimators_supported, metrics='mse', data_type='synthetic',
                 random_state=None, i_task=0, partial_res_file_name_root='./slope', stratify=True):
        super().__init__(ope_estimators, q_models, metrics, data_type, random_state, i_task, partial_res_file_name_root,
                         stratify)
        self.backend = 'threads'
        self.estimators_supported = estimators_supported  # Need to be sorted by variance
        self.rng = check_random_state(self.random_state + self.i_task)



    def evaluate_estimators_single_bootstrap_iter(self, log_data, pi_e_dist):
        """For given batch data, we estimate estimator performance

        Args:
            log_data (dict): batch bandit feedback
            pi_e_dist (np.array): action distribution by evaluation policy

        Returns:
            dict: key:estimator name, value:metric of estimator performance
        """
        all_candidate_estimators = []
        self.accumulated_intervals = []
        self.accumulated_means = []
        fulfills_condition = True

        for i, q_model in enumerate(self.q_models):
            for j, ope_estimator in enumerate(deepcopy(self.ope_estimators)):
                if 'IPW' not in ope_estimator.estimator_name:
                    ope_estimator.estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                    all_candidate_estimators.append(ope_estimator)
                else:
                    if i == 0:
                        all_candidate_estimators.append(ope_estimator)
        all_candidate_estimators_names = [oe.estimator_name for oe in all_candidate_estimators]
#
        #        if fulfills_condition:
        #            fulfills_condition, mean, interval = self.slope_overlaps(ope_estimator, log_data, pi_e_dist,
        #                                                                     estimated_rewards, self.accumulated_intervals)
        #            if fulfills_condition:
        #                curr_best_indices = j
        #            self.accumulated_intervals.append(pd.Interval(*interval))
        #            self.accumulated_means.append(mean)

        # SLOPE cannot distinguish between different reward models, so one is chosen at random
        q_model = self.rng.choice(self.q_models)
        estimated_rewards = self.get_estimated_rewards(log_data, q_model)
        all_supported_estimators_names = []

        for j, ope_estimator in enumerate(deepcopy(self.estimators_supported)):
            if 'IPW' not in ope_estimator.estimator_name:
                ope_estimator.estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
            all_supported_estimators_names.append(ope_estimator.estimator_name)

            if ope_estimator.estimator_name in all_supported_estimators_names:
                if fulfills_condition:
                    fulfills_condition, mean, interval = self.slope_overlaps(ope_estimator, log_data, pi_e_dist,
                                                                             estimated_rewards, self.accumulated_intervals)
                    if fulfills_condition:
                        curr_best_indices = j
                    self.accumulated_intervals.append(pd.Interval(*interval))
                    self.accumulated_means.append(mean)

        # TODO: create metric to handle spearman
        rank = [3] * len(all_candidate_estimators)
        for i, name in enumerate(all_candidate_estimators_names):
            if name in all_supported_estimators_names:
                rank[i] = 1 if name in [all_supported_estimators_names[curr_best_indices]] else 2

        mean_estimator_performance_df = pd.DataFrame({
            'estimator_name': all_candidate_estimators_names,
            self.metrics: rank
        })

        return self.get_single_bootstrap_performance(mean_estimator_performance_df)



    def _hyperparameters_tuning(self):
        return



    def slope_(self, ope_estimator, log_data, pi_e_dist, estimated_rwd, accumulated_means, accumulated_intervals):
        C = np.sqrt(6) - 1
        fullfils_condition = True
        mean, ci = self.get_estimator_mean_ci_given_rwd(ope_estimator, log_data, pi_e_dist, estimated_rwd)
        if len(accumulated_intervals) > 0:
            fullfils_condition = np.abs(mean - accumulated_means[-1]) <= ci[0] + C * accumulated_intervals[0][-1]
        return fullfils_condition, mean, ci



    def slope_overlaps(self, ope_estimator, log_data, pi_e_dist, estimated_rwd, accumulated_intervals):
        fullfils_condition = True
        # Clipped IPS or DR are not supported by SLOPE (no tuning)
        mean, var = self.get_estimator_mean_var_given_rwd(ope_estimator, log_data, pi_e_dist, estimated_rwd, tune=False)
        ci = (mean - 2 * np.sqrt(var), mean + 2 * np.sqrt(var))
        for other_interval in accumulated_intervals:
            if fullfils_condition:
                fullfils_condition = pd.Interval(*ci).overlaps(other_interval)
        return fullfils_condition, mean, ci



    def get_estimator_mean_ci_given_rwd(self, ope_estimator, log_data, pi_e_dist, estimated_rwd):
        round_rwards = ope_estimator._estimate_round_rewards(reward=log_data['reward'],
                                                             action=log_data['action'],
                                                             pscore=log_data['pscore'],
                                                             action_dist=pi_e_dist,
                                                             estimated_rewards_by_reg_model=estimated_rwd,
                                                             position=log_data['position'])
        mean = round_rwards.mean()
        width_ci = estimate_student_t_lower_bound(x=round_rwards, delta=0.1)
        cnf_low = mean - width_ci
        cnf_up = mean + width_ci
        return mean, (cnf_low, cnf_up)