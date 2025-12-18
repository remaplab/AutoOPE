from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats

from common.data.obp_data_train_test_split import ObpDataTrainTestSplit
from common.estimator_selection.base_estimator_selection import BaseEstimatorSelection



class OCVEstimatorSelection(BaseEstimatorSelection):
    """
    Off-Policy Cross-Validation estimator selection method
    """



    def __init__(self, ope_estimators, q_models, metrics='mse', data_type='synthetic', random_state=None, i_task=0,
                 partial_res_file_name_root='./ocv', stratify=True):
        super().__init__(ope_estimators, q_models, metrics, data_type, random_state, i_task, partial_res_file_name_root,
                         stratify)



    def set_ocv_params(self, valid_estimator, valid_q_model=None, K=10, train_ratio='theory', one_stderr_rule=True):
        self.K = K
        self.one_standard_error_rule = one_stderr_rule
        self.valid_estimator = valid_estimator
        self.valid_q_model = valid_q_model
        self.train_valid_ratio = train_ratio



    def evaluate_estimators_single_bootstrap_iter(self, log_data, pi_e_dist):
        """For given batch data, we estimate estimator performance

        Args:
            log_data (dict): batch bandit feedback
            pi_e_dist (np.array): action distribution by evaluation policy

        Returns:
            dict: key:estimator name, value:metric of estimator performance
        """
        errors = defaultdict(list)
        estimated_rewards_val = self.get_estimated_rewards(log_data, self.valid_q_model)
        _, valid_var = self.get_estimator_mean_var_given_rwd(self.valid_estimator, log_data, pi_e_dist,
                                                             estimated_rewards_val)
        for i, q_model in enumerate(self.q_models):
            estimated_rewards = self.get_estimated_rewards(log_data, q_model)

            for ope_estimator in deepcopy(self.ope_estimators):
                if (not "IPW" in ope_estimator.estimator_name) or i == 0 :
                    if 'IPW' not in ope_estimator.estimator_name:
                        ope_estimator.estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__

                    _, candidate_var = self.get_estimator_mean_var_given_rwd(ope_estimator, log_data, pi_e_dist,
                                                                             estimated_rewards)
                    train_ratio = candidate_var / (candidate_var + valid_var) if ((candidate_var + valid_var) != 0) else 0.5
                    test_ratio = 1 - train_ratio

                    for k in range(self.K):
                        stratify = log_data['reward']

                        # To have at least 1 sample for train and 1 sample for test with the same class distribution
                        if log_data['position'] is not None:
                            stratify = log_data['reward'] * (log_data['position'].max() + 1)  # reward in {0, max_position}
                            stratify += log_data['position']  # reward in {0, ..., 2 * max_position - 1}

                        classes, y_indices = np.unique(stratify, return_inverse=True)
                        class_counts = np.bincount(y_indices)
                        min_samples = int(np.ceil((class_counts / min(class_counts)).sum()))

                        if test_ratio * log_data['n_rounds'] < min_samples:
                            test_ratio = min_samples

                        if train_ratio * log_data['n_rounds'] < min_samples:
                            train_ratio = min_samples
                            test_ratio = log_data['n_rounds'] - min_samples

                        splitter = ObpDataTrainTestSplit(batch_bandit_feedback=log_data)
                        splitter.set_params(test_size=test_ratio, random_state=self.random_state + k, stratify=stratify)
                        log_data_train, log_data_val = splitter.get_train_test_data()
                        train_idx, val_idx = splitter.get_train_test_index()
                        pi_e_train, pi_e_val = pi_e_dist[train_idx], pi_e_dist[val_idx]

                        pi_e_value = self.get_policy_value_estimate(self.valid_estimator, q_model, log_data_val, pi_e_val)
                        pi_e_value_hat = self.get_policy_value_estimate(ope_estimator, q_model, log_data_train, pi_e_train)

                        errors[ope_estimator.estimator_name].append((pi_e_value_hat - pi_e_value) ** 2)

        result = pd.DataFrame(errors)
        if self.one_standard_error_rule:
            result = result.mean() + result.std() / result.count().map(np.sqrt)
        else:
            result = result.mean()
        result = result.to_dict()

        mean_estimator_performance_df = pd.DataFrame({'estimator_name': result.keys(),
                                                      self.metrics: result.values()})

        return self.get_single_bootstrap_performance(mean_estimator_performance_df)



    def _hyperparameters_tuning(self):
        pass

