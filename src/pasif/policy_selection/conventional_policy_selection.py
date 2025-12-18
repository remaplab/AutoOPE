# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import os
import random
import sys
from copy import deepcopy

import numpy as np
import obp
import pandas as pd
from joblib import Parallel, delayed
from obp.ope import OffPolicyEvaluation
from scipy import stats

from common.policy_selection.abstract_policy_selection import AbstractPolicySelection
from common.regression_model_stratified import RegressionModelStratified

if os.path.dirname(__file__) == '':
    sys.path.append('../../src')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from common.data.modify_batch_feedback import merge_batch_feedback
from common.data.counterfactual_pi_b import get_counterfactual_action_distribution
from common.data.obp_data_train_test_split import ObpDataTrainTestSplit
from pasif.estimator_selection.conventional_estimator_selection import ConventionalEstimatorSelection


class ConventionalPolicySelection(AbstractPolicySelection):
    """
    Conventional off-policy policy selection method
    """

    def __init__(self, ope_estimators, q_models, stratify, estimator_selection_metrics='mse', data_type='synthetic',
                 random_state=None):
        """
        Set basic settings

        Args:
            ope_estimators (list): list of candidate estimators
            q_models (list): list of reward estimators used in model-depending estimators
            estimator_selection_metrics (str, optional): Defaults to 'mse'. Must be 'mse' or 'mean relative-ee'
            data_type (str, optional): Defaults to 'synthetic'. Must be 'synthetic' or 'real'
            random_state (int, optional): Defaults to None.
        """
        super().__init__(ope_estimators, q_models, stratify, estimator_selection_metrics, data_type, random_state)

        self.original_data_seed_2 = None
        self.original_data_seed_1 = None
        self.pi_e = None
        self.n_rounds_2 = None
        self.dataset_2 = None
        self.n_rounds_1 = None
        self.dataset_1 = None
        self.evaluation_data = None
        self.action_dist_2_by_pi_e = None
        self.action_dist_1_by_pi_e = None
        self.action_dist_2_by_1 = None
        self.batch_bandit_feedback_2 = None
        self.action_dist_1_by_2 = None
        self.batch_bandit_feedback_1 = None

    def set_real_data(self, batch_bandit_feedback_1, action_dist_1_by_2, batch_bandit_feedback_2, action_dist_2_by_1,
                      action_dist_1_by_pi_e, action_dist_2_by_pi_e, evaluation_data='partial_random'):
        """
        Set real-world data (logged bandit feedback)

        Args:
            batch_bandit_feedback_1 (dict): batch bandit feedback 1
            action_dist_1_by_2 (np.array): action distribution for data 1 by policy 2
            batch_bandit_feedback_2 (dict): batch bandit feedback 2
            action_dist_2_by_1 (np.array): action distribution for data 2 by policy 1
            action_dist_1_by_pi_e (dict): dictionary of evaluation policies.
                keys: name of policy (str). values: array of pi_e for data 1.
            action_dist_2_by_pi_e (dict): dictionary of evaluation policies.
                keys: name of policy (str). values: array of pi_e for data 2.
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random' or 'partial_random'. 
                                             Which data (behavior policy) to consider as evaluation policy.
                                             'partial_random' means that we use fixed data as evalu policy in bootstrap,
                                             but no fixed in outer loop in estimator selection.
        """
        assert action_dist_1_by_pi_e.keys() == action_dist_2_by_pi_e.keys(), 'action_dist_1_by_pi_e and ' \
                                                                             'action_dist_2_by_pi_e must have same keys'
        self.batch_bandit_feedback_1 = batch_bandit_feedback_1
        self.action_dist_1_by_2 = action_dist_1_by_2
        self.batch_bandit_feedback_2 = batch_bandit_feedback_2
        self.action_dist_2_by_1 = action_dist_2_by_1
        self.action_dist_1_by_pi_e = action_dist_1_by_pi_e
        self.action_dist_2_by_pi_e = action_dist_2_by_pi_e
        self.evaluation_data = evaluation_data

    def set_synthetic_data(self, dataset_1, n_rounds_1, dataset_2, n_rounds_2, pi_e, evaluation_data='partial_random'):
        """
        Set synthetic data

        Args:
            dataset_1 (obp.dataset.SyntheticBanditDataset): synthetic data generator 1
            n_rounds_1 (int): sample size of data 1
            dataset_2 (obp.dataset.SyntheticBanditDataset): synthetic data generator 2
            n_rounds_2 (int): sample size of data 2
            pi_e (dict): dictionary of evaluation policies. keys: name of policy (str). values: tuple including info of
            evaluation policy.
                          ex. {'beta_1.0':('beta', 1.0), 'function_1':('function', pi(a|x), tau)} 
                          * ('beta', 1.0) (using beta to specify evaluation policy)
                          * ('function', pi(a|x), tau) (Give any function as evaluation policy. To get action_dist, we
                          use predict_proba(tau=tau))
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random' or 'partial_random'. 
                                             Which data (behavior policy) to consider as evaluation policy.
                                             'partial_random' means that we use fixed data as evalu policy in bootstrap,
                                             but no fixed in outer loop in estimator selection.
        """
        assert dataset_1.random_state == dataset_2.random_state, 'recommend ' \
                                                                 'dataset_1.random_state == dataset_2.random_state'
        self.dataset_1 = dataset_1
        self.n_rounds_1 = n_rounds_1
        self.dataset_2 = dataset_2
        self.n_rounds_2 = n_rounds_2
        self.pi_e = pi_e
        self.evaluation_data = evaluation_data
        self.original_data_seed_1 = self.dataset_1.random_state
        self.original_data_seed_2 = self.dataset_2.random_state

    def evaluate_policies_single_outer_loop(self, batch_bandit_feedback_1_train, action_dist_1_by_2_train,
                                            batch_bandit_feedback_2_train, action_dist_2_by_1_train,
                                            batch_bandit_feedback_1_test, action_dist_1_by_2_test,
                                            batch_bandit_feedback_2_test, action_dist_2_by_1_test,
                                            action_dist_1_by_pi_e_test, action_dist_2_by_pi_e_test, i_out,
                                            evaluation_data='random', n_bootstrap=100):
        """
        For given batch data and evaluation policies, we estimate estimator performance with bootstrap and select
        best policy

        Args:
            batch_bandit_feedback_1_train (dict): batch bandit feedback 1 for train (estimator selection)
            action_dist_1_by_2_train (np.array): action distribution for data 1 by policy 2 (train data for estimator
            selection)
            batch_bandit_feedback_2_train (dict): batch bandit feedback 2 for train (estimator selection)
            action_dist_2_by_1_train (np.array): action distribution for data 2 by policy 1 (train data for estimator
            selection)
            batch_bandit_feedback_1_test (dict): batch bandit feedback 1 for test (ope and policy selection)
            action_dist_1_by_2_test (np.array): action distribution for data 1 by policy 2 (test data for policy
            selection)
            batch_bandit_feedback_2_test (dict): batch bandit feedback 2 for test (ope and policy selection)
            action_dist_2_by_1_test (np.array): action distribution for data 2 by policy 1 (test data for policy
            selection)
            action_dist_1_by_pi_e_test (dict): dictionary of evaluation policies.
                keys: name of policy (str). values: array of pi_e for test data 1.
            action_dist_2_by_pi_e_test (dict): dictionary of evaluation policies.
                keys: name of policy (str). values: array of pi_e for test data 2.
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random'. Which data (behavior
            policy) to consider as evaluation policy in estimator selection.
            n_bootstrap (int, optional): Defaults to 100. The number of bootstrap sampling in estimator selection.
            If None, we use original data only once.
            i_out: index of the outer loop iteration
        Returns:
            pd.DataFrame, dict: dataframe with columns=[policy_name, estimator_name, estimated_policy_value, rank], and
            estimator selection result for each evaluation policy
        """

        # key:policy_name, value:df (estimator_name, mean mse, rank, estimated_policy_value_for_test_data)
        estimator_selection_result = {}
        policy_performance = pd.DataFrame(columns=['policy_name', 'estimator_name', 'estimated_policy_value'])

        # estimator selection
        conventional_estimator_selection = ConventionalEstimatorSelection(
            ope_estimators=self.ope_estimators,
            q_models=self.q_models,
            metrics=self.estimator_selection_metrics,
            data_type='real',
            random_state=self.random_state,
            stratify=self.stratify
        )
        conventional_estimator_selection.set_real_data(
            batch_bandit_feedback_1=batch_bandit_feedback_1_train,
            action_dist_1_by_2=action_dist_1_by_2_train,
            batch_bandit_feedback_2=batch_bandit_feedback_2_train,
            action_dist_2_by_1=action_dist_2_by_1_train,
            evaluation_data=evaluation_data
        )
        conventional_estimator_selection.evaluate_estimators(
            n_inner_bootstrap=n_bootstrap,
            n_outer_bootstrap=None,
            outer_repeat_type='bootstrap',
            ground_truth_method='on_policy'
        )
        summarized_estimator_selection_result = conventional_estimator_selection.get_summarized_results()[
            ['estimator_name', 'mean ' + self.estimator_selection_metrics, 'rank']]
        best_estimator, best_q_model = conventional_estimator_selection.get_best_estimator(np.random.RandomState(i_out))
        if 'IPW' in best_estimator.estimator_name:
            best_estimator_name = best_estimator.estimator_name
        else:
            best_estimator_name = best_estimator.estimator_name + '_qmodel_' + best_q_model.__name__

        # estimate policy value
        # merge data
        merged_test_data = merge_batch_feedback(
            batch_bandit_feedback_1=batch_bandit_feedback_1_test,
            action_dist_1_by_2=action_dist_1_by_2_test,
            batch_bandit_feedback_2=batch_bandit_feedback_2_test,
            action_dist_2_by_1=action_dist_2_by_1_test,
        )

        for policy_name in action_dist_1_by_pi_e_test.keys():
            merged_action_dist_by_pi_e = np.concatenate(
                [action_dist_1_by_pi_e_test[policy_name], action_dist_2_by_pi_e_test[policy_name]])
            estimated_policy_value = {}

            for i, q_model in enumerate(self.q_models):
                renamed_ope_estimators = []

                for ope_estimator in deepcopy(self.ope_estimators):
                    if 'IPW' in ope_estimator.estimator_name:
                        if i == 0:
                            renamed_ope_estimators.append(ope_estimator)
                        else:
                            pass
                    else:
                        new_estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                        ope_estimator.estimator_name = new_estimator_name
                        renamed_ope_estimators.append(ope_estimator)

                q_model_instance = self.get_q_model_instance(merged_test_data['n_rounds'], q_model)
                len_list = 1 if merged_test_data['position'] is None else int(merged_test_data['position'].max() + 1)

                # conduct ope for each evaluation policy
                regression_model = RegressionModelStratified(
                    n_actions=merged_test_data['n_actions'],
                    action_context=merged_test_data['action_context'],
                    base_model=q_model_instance,
                    len_list=len_list,
                    fitting_method='normal',
                    stratify=self.stratify
                )
                estimated_rewards_by_reg_model = regression_model.fit_predict(
                    context=merged_test_data["context"],
                    action=merged_test_data["action"],
                    reward=merged_test_data["reward"],
                    pscore=merged_test_data["pscore"],
                    position=merged_test_data['position'],
                    action_dist=merged_test_data['pi_b'],
                    n_folds=3,  # use 3-fold cross-fitting
                    random_state=self.random_state
                )

                ope = OffPolicyEvaluation(bandit_feedback=merged_test_data, ope_estimators=renamed_ope_estimators)
                estimated_policy_value.update(ope.estimate_policy_values(
                    action_dist=merged_action_dist_by_pi_e,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
                ))

            policy_performance = pd.concat([policy_performance, pd.DataFrame.from_dict({
                'policy_name': [policy_name],
                'estimator_name': [best_estimator_name],
                'estimated_policy_value': [estimated_policy_value[best_estimator_name]]})], ignore_index=True)

            estimated_policy_value = pd.DataFrame(
                {'estimator_name': estimated_policy_value.keys(),
                 'estimated_policy_value_for_test_data': estimated_policy_value.values()})
            estimator_selection_result[policy_name] = pd.merge(deepcopy(summarized_estimator_selection_result),
                                                               estimated_policy_value, how='left', on='estimator_name')

        # add rank
        policy_rank = policy_performance.rank(method='min', ascending=False)[['estimated_policy_value']].rename(
            columns={'estimated_policy_value': 'rank'})
        policy_performance = pd.merge(policy_performance, policy_rank, left_index=True, right_index=True)

        return policy_performance, estimator_selection_result

    def evaluate_policies(self, n_inference_bootstrap=1, n_task=10, test_ratio=0.5, outer_n_jobs=-1, inner_n_jobs=-1):
        """
        For set data, we evaluate policies (with bootstrapping estimator selection) for several times

        Args:
            n_inference_bootstrap (int): The number of bootstrap sampling in ope estimator selection. If None, we use
            original data only once.
            n_task (int, optional): Defaults to 10. The number of policy selection.
            test_ratio (float, optional): Defaults to 0.5. Ratio of test set. This is valid when real data are used.
            outer_n_jobs (int): number of concurrent jobs in outer loop
            inner_n_jobs (int): number of concurrent jobs in inner loop
        """
        outer_loop_type = 'uniform_sampling'
        
        if self.data_type == 'synthetic':
            outer_loop_type = 'generation'
            self.dataset_2.obtain_batch_bandit_feedback(
                n_rounds=self.n_rounds_2)  # to get different features between data 1 and 2

            self.action_dist_1_by_pi_e = {}
            self.action_dist_2_by_pi_e = {}

            for policy_name, pi_e in self.pi_e.items():
                if pi_e[0] == 'beta':
                    self.action_dist_1_by_pi_e[policy_name] = get_counterfactual_action_distribution(
                        dataset=self.dataset_1, cf_beta=pi_e[1], n_rounds=self.n_rounds_1)
                    self.action_dist_2_by_pi_e[policy_name] = get_counterfactual_action_distribution(
                        dataset=self.dataset_2, cf_beta=pi_e[1], n_rounds=self.n_rounds_2)

            self.action_dist_1_by_2 = get_counterfactual_action_distribution(dataset=self.dataset_1,
                                                                             cf_beta=self.dataset_2.beta,
                                                                             n_rounds=self.n_rounds_1)
            self.batch_bandit_feedback_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
            self.action_dist_2_by_1 = get_counterfactual_action_distribution(dataset=self.dataset_2,
                                                                             cf_beta=self.dataset_1.beta,
                                                                             n_rounds=self.n_rounds_2)
            self.batch_bandit_feedback_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

            for policy_name, pi_e in self.pi_e.items():
                if pi_e[0] == 'function':
                    action_dist_1_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_1['context'], tau=pi_e[2])
                    self.action_dist_1_by_pi_e[policy_name] = action_dist_1_by_pi_e.reshape(
                        (action_dist_1_by_pi_e.shape[0], action_dist_1_by_pi_e.shape[1], 1))
                    action_dist_2_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_2['context'], tau=pi_e[2])
                    self.action_dist_2_by_pi_e[policy_name] = action_dist_2_by_pi_e.reshape(
                        (action_dist_2_by_pi_e.shape[0], action_dist_2_by_pi_e.shape[1], 1))

        batch_bandit_feedback_1_list_train, batch_bandit_feedback_1_list_test = [], []
        action_dist_1_by_2_list_train, action_dist_1_by_2_list_test = [], []
        batch_bandit_feedback_1_train, batch_bandit_feedback_1_test = None, None
        action_dist_1_by_2_train, action_dist_1_by_2_test = {}, {}
        batch_bandit_feedback_2_list_train, batch_bandit_feedback_2_list_test = [], []
        action_dist_2_by_1_list_train, action_dist_2_by_1_list_test = [], []
        batch_bandit_feedback_2_train, batch_bandit_feedback_2_test = None, None
        action_dist_2_by_1_train, action_dist_2_by_1_test = {}, {}
        action_dist_1_by_pi_e_list, action_dist_2_by_pi_e_list = [], []
        action_dist_1_by_pi_e, action_dist_2_by_pi_e = {}, {}

        for idx in range(n_task):
            if outer_loop_type == 'uniform_sampling':
                train_test_split = ObpDataTrainTestSplit(self.batch_bandit_feedback_1)
                train_test_split.set_params(test_size=test_ratio, random_state=self.random_state + idx,
                                            stratify=self.batch_bandit_feedback_1['reward'])
                indices_train, indices_test = train_test_split.get_train_test_index()
                batch_bandit_feedback_1_train, batch_bandit_feedback_1_test = train_test_split.get_train_test_data()
                action_dist_1_by_2_train = self.action_dist_1_by_2[indices_train]
                action_dist_1_by_2_test = self.action_dist_1_by_2[indices_test]
                for policy_name, action_dist in self.action_dist_1_by_pi_e.items():
                    action_dist_1_by_pi_e[policy_name] = action_dist[indices_test]

                train_test_split = ObpDataTrainTestSplit(self.batch_bandit_feedback_2)
                train_test_split.set_params(test_size=test_ratio, random_state=self.random_state + idx,
                                            stratify=self.batch_bandit_feedback_2['reward'])
                indices_train, indices_test = train_test_split.get_train_test_index()
                batch_bandit_feedback_2_train, batch_bandit_feedback_2_test = train_test_split.get_train_test_data()
                action_dist_2_by_1_train = self.action_dist_2_by_1[indices_train]
                action_dist_2_by_1_test = self.action_dist_2_by_1[indices_test]
                for policy_name, action_dist in self.action_dist_1_by_pi_e.items():
                    action_dist_2_by_pi_e[policy_name] = action_dist[indices_test]

            elif outer_loop_type == 'generation':
                self.action_dist_1_by_pi_e = {}
                self.action_dist_2_by_pi_e = {}

                for policy_name, pi_e in self.pi_e.items():
                    if pi_e[0] == 'beta':
                        self.action_dist_1_by_pi_e[policy_name] = get_counterfactual_action_distribution(
                            dataset=self.dataset_1, cf_beta=pi_e[1], n_rounds=self.n_rounds_1)
                        self.action_dist_2_by_pi_e[policy_name] = get_counterfactual_action_distribution(
                            dataset=self.dataset_2, cf_beta=pi_e[1], n_rounds=self.n_rounds_2)

                self.action_dist_1_by_2 = get_counterfactual_action_distribution(dataset=self.dataset_1,
                                                                                 cf_beta=self.dataset_2.beta,
                                                                                 n_rounds=self.n_rounds_1)
                self.batch_bandit_feedback_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
                self.action_dist_2_by_1 = get_counterfactual_action_distribution(dataset=self.dataset_2,
                                                                                 cf_beta=self.dataset_1.beta,
                                                                                 n_rounds=self.n_rounds_2)
                self.batch_bandit_feedback_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

                for policy_name, pi_e in self.pi_e.items():
                    if pi_e[0] == 'function':
                        action_dist_1_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_1['context'],
                                                                      tau=pi_e[2])
                        self.action_dist_1_by_pi_e[policy_name] = action_dist_1_by_pi_e.reshape(
                            (action_dist_1_by_pi_e.shape[0], action_dist_1_by_pi_e.shape[1], 1))
                        action_dist_2_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_2['context'],
                                                                      tau=pi_e[2])
                        self.action_dist_2_by_pi_e[policy_name] = action_dist_2_by_pi_e.reshape(
                            (action_dist_2_by_pi_e.shape[0], action_dist_2_by_pi_e.shape[1], 1))
                if test_ratio is None:
                    batch_bandit_feedback_1_train = self.batch_bandit_feedback_1
                    batch_bandit_feedback_1_test = self.batch_bandit_feedback_1
                    action_dist_1_by_2_train = self.action_dist_1_by_2
                    action_dist_1_by_2_test = self.action_dist_1_by_2
                    batch_bandit_feedback_2_train = self.batch_bandit_feedback_2
                    batch_bandit_feedback_2_test = self.batch_bandit_feedback_2
                    action_dist_2_by_1_train = self.action_dist_2_by_1
                    action_dist_2_by_1_test = self.action_dist_2_by_1
                    action_dist_1_by_pi_e = self.action_dist_1_by_pi_e
                    action_dist_2_by_pi_e = self.action_dist_2_by_pi_e
                else:
                    train_test_split = ObpDataTrainTestSplit(self.batch_bandit_feedback_1)
                    train_test_split.set_params(test_size=test_ratio, random_state=self.random_state + idx,
                                                stratify=self.batch_bandit_feedback_1['reward'])
                    indices_train, indices_test = train_test_split.get_train_test_index()
                    batch_bandit_feedback_1_train, batch_bandit_feedback_1_test = train_test_split.get_train_test_data()
                    action_dist_1_by_2_train = self.action_dist_1_by_2[indices_train]
                    action_dist_1_by_2_test = self.action_dist_1_by_2[indices_test]
                    for policy_name, action_dist in self.action_dist_1_by_pi_e.items():
                        action_dist_1_by_pi_e[policy_name] = action_dist[indices_test]

                    train_test_split = ObpDataTrainTestSplit(self.batch_bandit_feedback_2)
                    train_test_split.set_params(test_size=test_ratio, random_state=self.random_state + idx,
                                                stratify=self.batch_bandit_feedback_1['reward'])
                    indices_train, indices_test = train_test_split.get_train_test_index()
                    batch_bandit_feedback_2_train, batch_bandit_feedback_2_test = train_test_split.get_train_test_data()
                    action_dist_2_by_1_train = self.action_dist_2_by_1[indices_train]
                    action_dist_2_by_1_test = self.action_dist_2_by_1[indices_test]
                    for policy_name, action_dist in self.action_dist_2_by_pi_e.items():
                        action_dist_2_by_pi_e[policy_name] = action_dist[indices_test]

            batch_bandit_feedback_1_list_train.append(batch_bandit_feedback_1_train)
            batch_bandit_feedback_1_list_test.append(batch_bandit_feedback_1_test)
            action_dist_1_by_2_list_train.append(action_dist_1_by_2_train)
            action_dist_1_by_2_list_test.append(action_dist_1_by_2_test)
            batch_bandit_feedback_2_list_train.append(batch_bandit_feedback_2_train)
            batch_bandit_feedback_2_list_test.append(batch_bandit_feedback_2_test)
            action_dist_2_by_1_list_train.append(action_dist_2_by_1_train)
            action_dist_2_by_1_list_test.append(action_dist_2_by_1_test)
            action_dist_1_by_pi_e_list.append(action_dist_1_by_pi_e)
            action_dist_2_by_pi_e_list.append(action_dist_2_by_pi_e)

        parallel = Parallel(n_jobs=outer_n_jobs, verbose=True)
        results = parallel(
            delayed(parallelizable_evaluate_policies_single_outer_loop)(
                self, i_outer, n_inference_bootstrap, action_dist_1_by_pi_e_list[i_outer],
                action_dist_2_by_pi_e_list[i_outer], action_dist_1_by_2_list_train[i_outer],
                batch_bandit_feedback_1_list_train[i_outer], action_dist_2_by_1_list_train[i_outer],
                batch_bandit_feedback_2_list_train[i_outer], action_dist_1_by_2_list_test[i_outer],
                batch_bandit_feedback_1_list_test[i_outer], action_dist_2_by_1_list_test[i_outer],
                batch_bandit_feedback_2_list_test[i_outer])
            for i_outer in range(n_task))

        policy_performance_list = [tuple_[0] for tuple_ in results]
        estimator_selection_result_list = [tuple_[1] for tuple_ in results]
        self.all_estimator_selection_result = {}
        self.all_result = pd.concat(policy_performance_list)
        for estimator_selection_result in estimator_selection_result_list:
            for p_name, e_selection_df in estimator_selection_result.items():
                to_concat = [e_selection_df]
                if p_name in self.all_estimator_selection_result.keys():
                    to_concat = [self.all_estimator_selection_result[p_name]] + to_concat
                self.all_estimator_selection_result[p_name] = pd.concat(to_concat)

        self.all_result = self.all_result[
            ['outer_iteration', 'policy_name', 'estimator_name', 'estimated_policy_value', 'rank']]
        self.summarized_result = pd.DataFrame(
            columns=['policy_name', 'mean estimated_policy_value', 'stdev', '95%CI(upper)', '95%CI(lower)'])
        for policy_name in self.all_result['policy_name'].unique():
            summarized_result = [policy_name, self.all_result['estimated_policy_value'][
                self.all_result['policy_name'] == policy_name].mean()]
            if self.all_result['outer_iteration'].max() > 0:
                summarized_result.append(
                    self.all_result['estimated_policy_value'][self.all_result['policy_name'] == policy_name].std())
                t_dist = stats.t(loc=summarized_result[1],
                                 scale=stats.sem(self.all_result['estimated_policy_value'][
                                                     self.all_result['policy_name'] == policy_name]),
                                 df=len(self.all_result['estimated_policy_value'][
                                            self.all_result['policy_name'] == policy_name]) - 1)
                bottom, up = t_dist.interval(alpha=0.95)
                summarized_result.append(up)
                summarized_result.append(bottom)
            else:
                summarized_result.append(None)
                summarized_result.append(None)
                summarized_result.append(None)
            self.summarized_result = self.summarized_result.append({
                'policy_name': summarized_result[0],
                'mean estimated_policy_value': summarized_result[1],
                'stdev': summarized_result[2],
                '95%CI(upper)': summarized_result[3],
                '95%CI(lower)': summarized_result[4]
            }, ignore_index=True)

        policy_rank = self.summarized_result.rank(method='min', ascending=False)[
            ['mean estimated_policy_value']].rename(columns={'mean estimated_policy_value': 'rank'})
        self.summarized_result = pd.merge(self.summarized_result, policy_rank, left_index=True, right_index=True)


def parallelizable_evaluate_policies_single_outer_loop(conventional_ps, i_outer, n_inner_bootstrap,
                                                       action_dist_1_by_pi_e, action_dist_2_by_pi_e,
                                                       action_dist_1_by_2_train, batch_bandit_feedback_1_train,
                                                       action_dist_2_by_1_train,
                                                       batch_bandit_feedback_2_train, action_dist_1_by_2_test,
                                                       batch_bandit_feedback_1_test, action_dist_2_by_1_test,
                                                       batch_bandit_feedback_2_test):
    """
    Utility function used only to parallelize jobs easily
    """
    if conventional_ps.evaluation_data == 'partial_random':
        random.seed(conventional_ps.random_state + i_outer)
        evaluation_data = random.choice(['1', '2'])
    else:
        evaluation_data = conventional_ps.evaluation_data

    policy_performance, estimator_selection_result = conventional_ps.evaluate_policies_single_outer_loop(
        batch_bandit_feedback_1_train=batch_bandit_feedback_1_train,
        action_dist_1_by_2_train=action_dist_1_by_2_train,
        batch_bandit_feedback_2_train=batch_bandit_feedback_2_train,
        action_dist_2_by_1_train=action_dist_2_by_1_train,
        batch_bandit_feedback_1_test=batch_bandit_feedback_1_test,
        action_dist_1_by_2_test=action_dist_1_by_2_test,
        batch_bandit_feedback_2_test=batch_bandit_feedback_2_test,
        action_dist_2_by_1_test=action_dist_2_by_1_test,
        action_dist_1_by_pi_e_test=action_dist_1_by_pi_e,
        action_dist_2_by_pi_e_test=action_dist_2_by_pi_e,
        evaluation_data=evaluation_data,
        n_bootstrap=n_inner_bootstrap,
        i_out=i_outer
    )
    policy_performance['outer_iteration'] = i_outer
    for p_name, e_selection_df in estimator_selection_result.items():
        e_selection_df['outer_iteration'] = i_outer
        estimator_selection_result[p_name] = e_selection_df

    return policy_performance, estimator_selection_result
