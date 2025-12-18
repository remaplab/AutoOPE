# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
import os
import sys
from copy import deepcopy

import obp
import pandas as pd
from joblib import delayed, Parallel
from obp.ope import OffPolicyEvaluation
from scipy import stats

from common.estimator_selection.abstract_estimator_selection import AbstractEstimatorSelection
from common.regression_model_stratified import RegressionModelStratified

if os.path.dirname(__file__) == '':
    sys.path.append('../..')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from common.data.bootstrap_batch_feedback import sample_bootstrap_batch_bandit_feedback
from common.data.counterfactual_pi_b import get_counterfactual_action_distribution


class ConventionalEstimatorSelection(AbstractEstimatorSelection):
    """
    Conventional off-policy estimator selection method
    """

    def __init__(self, ope_estimators, q_models, metrics='mse', data_type='synthetic', random_state=None, stratify:
                 bool = True):
        """
        Set basic settings

        Args:
            ope_estimators (list): list of candidate estimators
            q_models (list): list of reward estimators used in model-depending estimators
            metrics (str, optional): Defaults to 'mse'. Must be 'mse' or 'mean relative-ee'
            data_type (str, optional): Defaults to 'synthetic'. Must be 'synthetic' or 'real'
            random_state (int, optional): Defaults to None.
        """

        super().__init__(ope_estimators, q_models, random_state)
        assert metrics == 'mse' or metrics == 'mean relative-ee', 'metrics must be mse or mean relative-ee'
        assert data_type == 'synthetic' or data_type == 'real', 'data_type must be synthetic or real'

        self.metrics = metrics
        self.data_type = data_type
        self.n_jobs_fit = 1
        self.stratify = stratify

        self.evaluation_data = None
        self.batch_bandit_feedback_1 = None
        self.action_dist_1_by_2 = None
        self.batch_bandit_feedback_2 = None
        self.action_dist_2_by_1 = None

        self.dataset_1 = None
        self.n_rounds_1 = None
        self.dataset_2 = None
        self.n_rounds_2 = None

    def set_real_data(
            self,
            batch_bandit_feedback_1,
            action_dist_1_by_2,
            batch_bandit_feedback_2,
            action_dist_2_by_1,
            evaluation_data='random'
    ):
        """
        Set real-world data (logged bandit feedback)

        Args:
            batch_bandit_feedback_1 (dict): batch bandit feedback 1
            action_dist_1_by_2 (np.array): action distribution for data 1 by policy 2
            batch_bandit_feedback_2 (dict): batch bandit feedback 2
            action_dist_2_by_1 (np.array): action distribution for data 2 by policy 1
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random'.
            Which data (behavior policy) to consider as evaluation policy.
        """
        self.batch_bandit_feedback_1 = batch_bandit_feedback_1
        self.action_dist_1_by_2 = action_dist_1_by_2
        self.batch_bandit_feedback_2 = batch_bandit_feedback_2
        self.action_dist_2_by_1 = action_dist_2_by_1
        self.evaluation_data = evaluation_data

    def set_synthetic_data(self, dataset_1, n_rounds_1, dataset_2, n_rounds_2, evaluation_data='random'):
        """
        Set synthetic data

        Args:
            dataset_1 (obp.dataset.SyntheticBanditDataset): synthetic data generator 1
            n_rounds_1 (int): sample size of data 1
            dataset_2 (obp.dataset.SyntheticBanditDataset): synthetic data generator 2
            n_rounds_2 (int): sample size of data 2
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random'.
            Which data (behavior policy) to consider as evaluation policy.
        """
        assert dataset_1.random_state == dataset_2.random_state, 'recommend dataset_1.random_state == ' \
                                                                 'dataset_2.random_state'
        self.dataset_1 = dataset_1
        self.n_rounds_1 = n_rounds_1
        self.dataset_2 = dataset_2
        self.n_rounds_2 = n_rounds_2
        self.evaluation_data = evaluation_data

    def _calculate_ground_truth(self, data_number=1, method='on-policy'):
        """
        Calculate the ground-truth of behavior policy in dataset 1

        Args:
            data_number (int, optional): Defaults to 1. Indicate whther we use data 1 or 2 to calculate the policy value
            method (str, optional): Defaults to 'on-policy'. Must be 'on-policy' or 'ground-truth'. 
                                    'on-policy' means on-policy estimation. 
                                    'ground-truth' means that we calculate the policy value using true expected reward
                                    and evaluation policy
        """
        if self.data_type == 'synthetic':
            dataset_dict = {1: self.dataset_1, 2: self.dataset_2}
        batch_bandit_feedback_dict = {1: self.batch_bandit_feedback_1, 2: self.batch_bandit_feedback_2}
        if method == 'on-policy':
            self.ground_truth = batch_bandit_feedback_dict[data_number]['reward'].mean()
        elif method == 'ground-truth':
            batch_with_large_n = dataset_dict[data_number].obtain_batch_bandit_feedback(n_rounds=1000000)
            self.ground_truth = dataset_dict[data_number].calc_ground_truth_policy_value(
                expected_reward=batch_with_large_n['expected_reward'],
                action_dist=batch_with_large_n['pi_b']
            )
        return self.ground_truth

    def evaluate_estimators_single_bootstrap_iter(
            self,
            batch_bandit_feedback_1,
            action_dist_1_by_2,
            batch_bandit_feedback_2,
            action_dist_2_by_1,
            ground_truth_method='on_policy'
    ):
        """For given batch data, we estimate estimator performance with bootstrap

        Args:
            batch_bandit_feedback_1 (dict): batch bandit feedback 1
            action_dist_1_by_2 (np.array): action distribution for data 1 by policy 2
            batch_bandit_feedback_2 (dict): batch bandit feedback 2
            action_dist_2_by_1 (np.array): action distribution for data 2 by policy 1
            ground_truth_method (str, optional): Defaults to 'on-policy'. Must be 'on-policy' or 'ground-truth'. 

        Returns:
            pd.DataFrame: dataframe with columns=[estimator_name, metrics, rank]
        """
        if self.evaluation_data == '1':
            mean_estimator_performance_dict = self.estimator_selection(action_dist_2_by_1, batch_bandit_feedback_1,
                                                                       batch_bandit_feedback_2, ground_truth_method)
        else:
            mean_estimator_performance_dict = self.estimator_selection(action_dist_1_by_2, batch_bandit_feedback_2,
                                                                       batch_bandit_feedback_1, ground_truth_method)

        mean_estimator_performance_df = pd.DataFrame({
            'estimator_name': mean_estimator_performance_dict.keys(),
            self.metrics: mean_estimator_performance_dict.values()
        })
        estimator_rank = mean_estimator_performance_df.rank(method='min')[[self.metrics]].rename(
            columns={self.metrics: 'rank'})
        mean_estimator_performance_df = pd.merge(mean_estimator_performance_df, estimator_rank, left_index=True,
                                                 right_index=True)
        return mean_estimator_performance_df

    def estimator_selection(self, pi_e_dist, eval_data, log_data, ground_truth_method):
        mean_estimator_performance_dict = {}
        estimator_performance = {}
        for i, q_model in enumerate(self.q_models):
            renamed_ope_estimators = []

            for ope_estimator in deepcopy(self.ope_estimators):
                if 'IPW' in ope_estimator.estimator_name or 'opera' in ope_estimator.estimator_name:
                    if i == 0:
                        renamed_ope_estimators.append(ope_estimator)
                else:
                    new_estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                    ope_estimator.estimator_name = new_estimator_name
                    renamed_ope_estimators.append(ope_estimator)

            q_model_instance = self.get_q_model_instance(log_data, q_model)
            q_model_instance.n_jobs = self.n_jobs_fit
            len_list = 1 if log_data['position'] is None else int(log_data['position'].max() + 1)

            regression_model = RegressionModelStratified(
                n_actions=log_data['n_actions'],
                action_context=log_data['action_context'],
                base_model=q_model_instance,
                len_list=len_list,
                fitting_method='normal',
                stratify=self.stratify
            )
            estimated_rewards_by_reg_model = regression_model.fit_predict(
                context=log_data['context'],
                action=log_data['action'],
                reward=log_data['reward'],
                pscore=log_data['pscore'],
                position=log_data['position'],
                action_dist=log_data['pi_b'],
                n_folds=3,  # use 3-fold cross-fitting
                random_state=self.random_state
            )
            ope = OffPolicyEvaluation(bandit_feedback=log_data, ope_estimators=renamed_ope_estimators)

            if self.metrics == 'mse':
                metric = 'se'
            elif self.metrics == 'mean relative-ee':
                metric = 'relative-ee'

            if ground_truth_method == 'on_policy':
                ground_truth = eval_data['reward'].mean()
            elif ground_truth_method == 'ground-truth':
                ground_truth = self._calculate_ground_truth(data_number=1, method='ground-truth')
            estimator_performance_for_q_model = ope.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth,
                action_dist=pi_e_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                pi_b=log_data['pi_b'],
                metric=metric,
            )
            estimator_performance.update(estimator_performance_for_q_model)
            mean_estimator_performance_dict.update(estimator_performance)
        return mean_estimator_performance_dict

    def evaluate_estimators(self, n_inner_bootstrap, n_outer_bootstrap, outer_repeat_type='bootstrap',
                            ground_truth_method='on_policy', n_jobs=-1):
        """for set data, we evaluate ope estimators with bootstrap for several times

        Args:
            n_inner_bootstrap (int): The number of bootstrap sampling in ope estimator selection. If None, we use
            original data only once.
            n_outer_bootstrap (int): The number of evaluation of estimators. If None, we use original data only once.
            outer_repeat_type (str, optional): Defaults to 'bootstrap'. If we use synthetic data, we can also use
            'generation' meaning generating data for each outer loop
            ground_truth_method (str, optional): Defaults to 'on_policy'. Must be 'on-policy' or 'ground-truth'.
            n_jobs: number of concurrent jobs
        """

        if n_jobs == 1:
            print('Single thread mode.')
            self.n_jobs_fit = -1
            return self.evaluate_estimators_single_thread(
                n_inner_bootstrap=n_inner_bootstrap, n_outer_bootstrap=n_outer_bootstrap,
                outer_repeat_type=outer_repeat_type, ground_truth_method=ground_truth_method
            )
        if self.data_type == 'synthetic':
            if type(self.dataset_2.beta) == list:
                assert self.evaluation_data == '1', 'if you use multibeta data, please set multibeta data ' \
                                                    'in dataset_2 and evaluation_data == 1'
                self.action_dist_1_by_2 = None
            else:
                self.action_dist_1_by_2 = get_counterfactual_action_distribution(dataset=self.dataset_1,
                                                                                 cf_beta=self.dataset_2.beta,
                                                                                 n_rounds=self.n_rounds_1)
            self.batch_bandit_feedback_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
            self.dataset_2.obtain_batch_bandit_feedback(
                n_rounds=self.n_rounds_2)  # to get different features between data 1 and 2
            self.action_dist_2_by_1 = get_counterfactual_action_distribution(dataset=self.dataset_2,
                                                                             cf_beta=self.dataset_1.beta,
                                                                             n_rounds=self.n_rounds_2)
            self.batch_bandit_feedback_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

        if n_outer_bootstrap is None:
            self.all_result = self.evaluate_estimators_single_bootstrap_iter(
                batch_bandit_feedback_1=self.batch_bandit_feedback_1,
                action_dist_1_by_2=self.action_dist_1_by_2,
                batch_bandit_feedback_2=self.batch_bandit_feedback_2,
                action_dist_2_by_1=self.action_dist_2_by_1,
                ground_truth_method=ground_truth_method
            )
            self.all_result['outer_iteration'] = 0
        else:
            bootstrapped_dist_1_list, bootstrapped_dist_2_list = [], []
            bootstrapped_data_1_list, bootstrapped_data_2_list = [], []
            bootstrapped_dist_1, bootstrapped_dist_2 = None, None
            bootstrapped_data_1, bootstrapped_data_2 = None, None
            for idx in range(n_outer_bootstrap):
                if outer_repeat_type == 'generation':
                    if type(self.dataset_2.beta) == list:
                        bootstrapped_dist_1 = None
                    else:
                        bootstrapped_dist_1 = get_counterfactual_action_distribution(dataset=self.dataset_1,
                                                                                     cf_beta=self.dataset_2.beta,
                                                                                     n_rounds=self.n_rounds_1)
                    bootstrapped_data_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
                    bootstrapped_dist_2 = get_counterfactual_action_distribution(dataset=self.dataset_2,
                                                                                 cf_beta=self.dataset_1.beta,
                                                                                 n_rounds=self.n_rounds_2)
                    bootstrapped_data_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

                elif outer_repeat_type == 'bootstrap':
                    bootstrapped_data_1, bootstrapped_dist_1 = sample_bootstrap_batch_bandit_feedback(
                        batch_bandit_feedback=self.batch_bandit_feedback_1,
                        action_dist=self.action_dist_1_by_2,
                        sample_size_ratio=1.0,
                        random_state=self.random_state + idx
                    )
                    bootstrapped_data_2, bootstrapped_dist_2 = sample_bootstrap_batch_bandit_feedback(
                        batch_bandit_feedback=self.batch_bandit_feedback_2,
                        action_dist=self.action_dist_2_by_1,
                        sample_size_ratio=1.0,
                        random_state=self.random_state + idx
                    )

                bootstrapped_dist_1_list.append(bootstrapped_dist_1)
                bootstrapped_dist_2_list.append(bootstrapped_dist_2)
                bootstrapped_data_1_list.append(bootstrapped_data_1)
                bootstrapped_data_2_list.append(bootstrapped_data_2)

            self.batch_bandit_feedback_1 = None
            self.action_dist_1_by_2 = None
            self.batch_bandit_feedback_2 = None
            self.action_dist_2_by_1 = None
            parallel = Parallel(n_jobs=n_jobs, verbose=True)
            results = parallel(
                delayed(parallelizable_evaluate_estimators_single_outer_loop)(
                    self, bootstrapped_data_1_list[i_outer],
                    bootstrapped_data_2_list[i_outer], bootstrapped_dist_1_list[i_outer],
                    bootstrapped_dist_2_list[i_outer], ground_truth_method, i_outer, n_inner_bootstrap)
                for i_outer in range(n_outer_bootstrap))
            i_outer_results = [res for res in results]
            self.all_result = pd.concat(i_outer_results)

        self.all_result = self.all_result[['outer_iteration', 'estimator_name', self.metrics, 'rank']]
        self.summarized_result = pd.DataFrame(
            columns=['estimator_name', 'mean ' + self.metrics, 'stdev', '95%CI(upper)', '95%CI(lower)'])
        for estimator_name in self.all_result['estimator_name'].unique():
            summarized_result = [estimator_name, self.all_result[self.metrics][
                self.all_result['estimator_name'] == estimator_name].mean()]
            if self.all_result['outer_iteration'].max() > 0:
                summarized_result.append(
                    self.all_result[self.metrics][self.all_result['estimator_name'] == estimator_name].std())
                t_dist = stats.t(loc=summarized_result[1],
                                 scale=stats.sem(self.all_result[self.metrics][
                                                     self.all_result['estimator_name'] == estimator_name]),
                                 df=len(self.all_result[self.metrics][
                                            self.all_result['estimator_name'] == estimator_name]) - 1)
                bottom, up = t_dist.interval(alpha=0.95)
                summarized_result.append(up)
                summarized_result.append(bottom)
            else:
                summarized_result.append(None)
                summarized_result.append(None)
                summarized_result.append(None)
            self.summarized_result = pd.concat([self.summarized_result, pd.DataFrame.from_dict({
                'estimator_name': [summarized_result[0]],
                'mean ' + self.metrics: [summarized_result[1]],
                'stdev': [summarized_result[2]],
                '95%CI(upper)': [summarized_result[3]],
                '95%CI(lower)': [summarized_result[4]]
            })], ignore_index=True)

        estimator_rank = self.summarized_result.rank(method='min')[['mean ' + self.metrics]].rename(
            columns={'mean ' + self.metrics: 'rank'})
        self.summarized_result = pd.merge(self.summarized_result, estimator_rank, left_index=True, right_index=True)

    def evaluate_estimators_single_thread(self, n_inner_bootstrap, n_outer_bootstrap, outer_repeat_type='bootstrap',
                                          ground_truth_method='on_policy'):
        """for set data, we evaluate ope estimators with bootstrap for several times

        Args:
            n_inner_bootstrap (int): The number of bootstrap sampling in ope estimator selection. If None, we use
            original data only once.
            n_outer_bootstrap (int): The number of evaluation of estimators. If None, we use original data only once.
            outer_repeat_type (str, optional): Defaults to 'bootstrap'. If we use synthetic data, we can also use
            'generation' meaning generating data for each outer loop
            ground_truth_method (str, optional): Defaults to 'on_policy'. Must be 'on-policy' or 'ground-truth'.
        """

        if self.data_type == 'synthetic':
            if type(self.dataset_2.beta) == list:
                assert self.evaluation_data == '1', 'if you use multibeta data, please set multibeta data ' \
                                                    'in dataset_2 and evaluation_data == 1'
                self.action_dist_1_by_2 = None
            else:
                self.action_dist_1_by_2 = get_counterfactual_action_distribution(dataset=self.dataset_1,
                                                                                 cf_beta=self.dataset_2.beta,
                                                                                 n_rounds=self.n_rounds_1)
            self.batch_bandit_feedback_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
            self.dataset_2.obtain_batch_bandit_feedback(
                n_rounds=self.n_rounds_2)  # to get different features between data 1 and 2
            self.action_dist_2_by_1 = get_counterfactual_action_distribution(dataset=self.dataset_2,
                                                                             cf_beta=self.dataset_1.beta,
                                                                             n_rounds=self.n_rounds_2)
            self.batch_bandit_feedback_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

        if n_outer_bootstrap is None:
            self.all_result = self.evaluate_estimators_single_bootstrap_iter(
                batch_bandit_feedback_1=self.batch_bandit_feedback_1,
                action_dist_1_by_2=self.action_dist_1_by_2,
                batch_bandit_feedback_2=self.batch_bandit_feedback_2,
                action_dist_2_by_1=self.action_dist_2_by_1,
                ground_truth_method=ground_truth_method
            )
            self.all_result['outer_iteration'] = 0
        else:
            bootstrapped_dist_1, bootstrapped_dist_2 = None, None
            bootstrapped_data_1, bootstrapped_data_2 = None, None
            results = []
            for idx in range(n_outer_bootstrap):
                if outer_repeat_type == 'generation':
                    if type(self.dataset_2.beta) == list:
                        bootstrapped_dist_1 = None
                    else:
                        bootstrapped_dist_1 = get_counterfactual_action_distribution(dataset=self.dataset_1,
                                                                                     cf_beta=self.dataset_2.beta,
                                                                                     n_rounds=self.n_rounds_1)
                    bootstrapped_data_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
                    bootstrapped_dist_2 = get_counterfactual_action_distribution(dataset=self.dataset_2,
                                                                                 cf_beta=self.dataset_1.beta,
                                                                                 n_rounds=self.n_rounds_2)
                    bootstrapped_data_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

                elif outer_repeat_type == 'bootstrap':
                    bootstrapped_data_1, bootstrapped_dist_1 = sample_bootstrap_batch_bandit_feedback(
                        batch_bandit_feedback=self.batch_bandit_feedback_1,
                        action_dist=self.action_dist_1_by_2,
                        sample_size_ratio=1.0,
                        random_state=self.random_state + idx
                    )
                    bootstrapped_data_2, bootstrapped_dist_2 = sample_bootstrap_batch_bandit_feedback(
                        batch_bandit_feedback=self.batch_bandit_feedback_2,
                        action_dist=self.action_dist_2_by_1,
                        sample_size_ratio=1.0,
                        random_state=self.random_state + idx
                    )

                res = parallelizable_evaluate_estimators_single_outer_loop(
                    self, bootstrapped_data_1, bootstrapped_data_2, bootstrapped_dist_1, bootstrapped_dist_2,
                    ground_truth_method, idx, n_inner_bootstrap)
                results.append(res)

            i_outer_results = [res for res in results]
            self.all_result = pd.concat(i_outer_results)

        self.all_result = self.all_result[['outer_iteration', 'estimator_name', self.metrics, 'rank']]
        self.summarized_result = pd.DataFrame(
            columns=['estimator_name', 'mean ' + self.metrics, 'stdev', '95%CI(upper)', '95%CI(lower)'])
        for estimator_name in self.all_result['estimator_name'].unique():
            summarized_result = [estimator_name, self.all_result[self.metrics][
                self.all_result['estimator_name'] == estimator_name].mean()]
            if self.all_result['outer_iteration'].max() > 0:
                summarized_result.append(
                    self.all_result[self.metrics][self.all_result['estimator_name'] == estimator_name].std())
                t_dist = stats.t(loc=summarized_result[1],
                                 scale=stats.sem(self.all_result[self.metrics][
                                                     self.all_result['estimator_name'] == estimator_name]),
                                 df=len(self.all_result[self.metrics][
                                            self.all_result['estimator_name'] == estimator_name]) - 1)
                bottom, up = t_dist.interval(alpha=0.95)
                summarized_result.append(up)
                summarized_result.append(bottom)
            else:
                summarized_result.append(None)
                summarized_result.append(None)
                summarized_result.append(None)
            self.summarized_result = pd.concat([self.summarized_result, pd.DataFrame.from_dict({
                'estimator_name': [summarized_result[0]],
                'mean ' + self.metrics: [summarized_result[1]],
                'stdev': [summarized_result[2]],
                '95%CI(upper)': [summarized_result[3]],
                '95%CI(lower)': [summarized_result[4]]
            })], ignore_index=True)

        estimator_rank = self.summarized_result.rank(method='min')[['mean ' + self.metrics]].rename(
            columns={'mean ' + self.metrics: 'rank'})
        self.summarized_result = pd.merge(self.summarized_result, estimator_rank, left_index=True, right_index=True)


def parallelizable_evaluate_estimators_single_outer_loop(
        conventional_es, bootstrapped_data_1, bootstrapped_data_2, bootstrapped_dist_1, bootstrapped_dist_2,
        ground_truth_method, i_outer, n_inner_bootstrap):
    i_outer_result = conventional_es.evaluate_estimators_single_bootstrap_iter(
        batch_bandit_feedback_1=bootstrapped_data_1,
        action_dist_1_by_2=bootstrapped_dist_1,
        batch_bandit_feedback_2=bootstrapped_data_2,
        action_dist_2_by_1=bootstrapped_dist_2,
        ground_truth_method=ground_truth_method
    )
    i_outer_result['outer_iteration'] = i_outer
    return i_outer_result
