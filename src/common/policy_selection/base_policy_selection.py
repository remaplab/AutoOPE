import os
import pickle
import time
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from obp.ope import OffPolicyEvaluation
from scipy import stats

from common.data.bootstrap_batch_feedback import subsample_batch_bandit_feedback
from common.data.counterfactual_pi_b import get_counterfactual_action_distribution
from common.data.obp_data_train_test_split import ObpDataTrainTestSplit
from common.policy_selection.abstract_policy_selection import AbstractPolicySelection
from common.regression_model_stratified import RegressionModelStratified


class BasePolicySelection(AbstractPolicySelection):
    """
    Base class for policy selection methods
    """

    def __init__(self, ope_estimators, q_models, stratify, estimator_selection_metrics='mse', data_type='synthetic',
                 random_state=None, log_dir='./', save=False):
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

        self.policy_selection_name = 'base'
        self.backend = 'processes'  # TODO: check here and in the subclasses (also in estimator selection classes!)
        self.timing = pd.DataFrame(index=[], columns=['iteration', 'policy_name', 'time_est_sel', 'time_ope'])

        self.log_dir = log_dir
        self.inner_n_jobs = None
        self.synth_n_rounds = None
        self.n_inference_bootstrap = None

        self.undersampling_ratio = None

        self.real_pi_e_distributions = {}
        self.real_log_data = None
        self.synth_dataset = None
        self.synth_pi_e_params = None

        self.save = save

    def set_real_data(self, log_data, pi_e_distributions, undersampling_ratio):
        """
        Set real-world data (logged bandit feedback)

        Args:
            log_data (dict): batch bandit feedback
            pi_e_distributions (dict): dictionary of evaluation policies.
                keys -> name of policy (str).
                values -> tuple with pi_e distribution for log data and info of evaluation policy.
                ex. ([[[0.1], ..., [0.1]], [[0.1], ..., [0.1]]], 10.0)
            undersampling_ratio (float): percentage of data to sample
        """
        self.real_log_data = log_data
        self.real_pi_e_distributions = pi_e_distributions
        self.undersampling_ratio = undersampling_ratio

    def set_synthetic_data(self, synth_dataset, n_rounds, pi_e_params):
        """
        Set synthetic data

        Args:
            synth_dataset (obp.dataset.SyntheticBanditDataset): synthetic data generator
            n_rounds (int): sample size of data
            pi_e_params (dict): dictionary of evaluation policies. keys: name of policy (str). values: tuple including info of
            evaluation policy.
              ex. {'beta_1.0':('beta', 1.0), 'function_1':('function', pi(a|x), tau)}
              * ('beta', 1.0) (using beta to specify evaluation policy)
              * ('function', pi(a|x), tau) (Give any function as evaluation policy. To get action_dist, we use
                predict_proba(tau=tau))
        """
        self.synth_dataset = synth_dataset
        self.synth_n_rounds = n_rounds
        self.synth_pi_e_params = pi_e_params

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
        self.inner_n_jobs = inner_n_jobs
        self.n_inference_bootstrap = n_inference_bootstrap

        if self.data_type == 'synthetic':
            pi_e_dist_train_list, pi_e_dist_test_list, log_data_train_list, log_data_test_list = \
                self.prepare_synthetic_data(n_task)
        else:
            pi_e_dist_train_list, pi_e_dist_test_list, log_data_train_list, log_data_test_list = \
                self.prepare_real_data(n_task, test_ratio)

        # TODO: set to None log_data and pi_e_distributions to reduce memory overhead? They don't need anymore
        self.real_log_data = None
        self.real_pi_e_distributions = None

        parallel = Parallel(n_jobs=outer_n_jobs, verbose=True, prefer=self.backend)
        raw_results = parallel(delayed(parallelizable_evaluate_policies_single_task)(
            self, i_task, n_inference_bootstrap,
            log_data_train_list[i_task] if log_data_train_list is not None else None,
            pi_e_dist_train_list[i_task] if pi_e_dist_train_list is not None else None,
            log_data_test_list[i_task] if log_data_test_list is not None else None,
            pi_e_dist_test_list[i_task] if pi_e_dist_test_list is not None else None
        ) for i_task in range(n_task))

        self.aggregate_results(raw_results)

    def aggregate_results(self, results):
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

        self.all_result = self.all_result[['outer_iteration', 'policy_name', 'estimator_name', 'estimated_policy_value',
                                           'rank']]
        self.summarized_result = pd.DataFrame(columns=['policy_name', 'mean estimated_policy_value', 'stdev',
                                                       '95%CI(upper)', '95%CI(lower)'])
        for policy_name in self.all_result['policy_name'].unique():
            summarized_result = [policy_name,
                                 self.all_result['estimated_policy_value'][
                                     self.all_result['policy_name'] == policy_name].mean()]
            if self.all_result['outer_iteration'].max() > 0:
                summarized_result.append(
                    self.all_result['estimated_policy_value'][self.all_result['policy_name'] == policy_name].std())
                t_dist = stats.t(
                    loc=summarized_result[1],
                    scale=stats.sem(
                        self.all_result['estimated_policy_value'][self.all_result['policy_name'] == policy_name]),
                    df=len(self.all_result['estimated_policy_value'][self.all_result['policy_name'] == policy_name]) - 1
                )
                bottom, up = t_dist.interval(alpha=0.95)
                summarized_result.append(up)
                summarized_result.append(bottom)
            else:
                summarized_result.append(None)
                summarized_result.append(None)
                summarized_result.append(None)

            self.summarized_result = pd.concat([self.summarized_result, pd.DataFrame.from_dict({
                'policy_name': [summarized_result[0]],
                'mean estimated_policy_value': [summarized_result[1]],
                'stdev': [summarized_result[2]],
                '95%CI(upper)': [summarized_result[3]],
                '95%CI(lower)': [summarized_result[4]]
            })], ignore_index=True)

        policy_rank = self.summarized_result.rank(
            method='min', ascending=False
        )[['mean estimated_policy_value']].rename(columns={'mean estimated_policy_value': 'rank'})

        self.summarized_result = pd.merge(self.summarized_result, policy_rank, left_index=True, right_index=True)

    def prepare_real_data(self, n_task, test_ratio):
        pi_e_dist_train_list, pi_e_dist_test_list, log_data_train_list, log_data_test_list = [], [], [], []
        pi_e_distributions = {}

        for policy_name, pi_e in self.real_pi_e_distributions.items():
            pi_e_distributions[policy_name] = pi_e[0]

        for i in range(n_task):
            us_log_data_i, us_pi_e_distributions_i = None, {}
            for policy_name, action_dist in pi_e_distributions.items():
                us_log_data_i, us_pi_e_distribution = subsample_batch_bandit_feedback(
                    batch_bandit_feedback=self.real_log_data,
                    action_dist=action_dist,
                    sample_size_ratio=self.undersampling_ratio,
                    random_state=self.random_state + i
                )
                us_pi_e_distributions_i[policy_name] = us_pi_e_distribution

            train_test_splitter_i = ObpDataTrainTestSplit(us_log_data_i)
            train_test_splitter_i.set_params(test_size=test_ratio, random_state=self.random_state + i,
                                             stratify=us_log_data_i['reward'])
            indices_train_i, indices_test_i = train_test_splitter_i.get_train_test_index()
            log_data_train_i, log_data_test_i = train_test_splitter_i.get_train_test_data()
            pi_e_dist_train_i, pi_e_dist_test_i = {}, {}
            for policy_name, action_dist in us_pi_e_distributions_i.items():
                pi_e_dist_train_i[policy_name] = action_dist[indices_train_i]
                pi_e_dist_test_i[policy_name] = action_dist[indices_test_i]

            pi_e_dist_train_list.append(pi_e_dist_train_i)
            log_data_train_list.append(log_data_train_i)
            pi_e_dist_test_list.append(pi_e_dist_test_i)
            log_data_test_list.append(log_data_test_i)

        return pi_e_dist_train_list, pi_e_dist_test_list, log_data_train_list, log_data_test_list

    def prepare_synthetic_data(self, n_task):
        pi_e_dist_train_list, pi_e_dist_test_list, log_data_train_list, log_data_test_list = [], [], [], []
        log_data_i, pi_e_distributions_i = None, {}

        for i_outer in range(n_task):
            for policy_name, pi_e_params in self.synth_pi_e_params.items():
                if pi_e_params[0] == 'beta':
                    pi_e_distributions_i[policy_name] = get_counterfactual_action_distribution(
                        dataset=self.synth_dataset, cf_beta=pi_e_params[1], n_rounds=self.synth_n_rounds)

            log_data_i = self.synth_dataset.obtain_batch_bandit_feedback(n_rounds=self.synth_n_rounds)

            for policy_name, pi_e_params in self.synth_pi_e_params.items():
                if pi_e_params[0] == 'function':
                    action_dist = pi_e_params[1].predict_proba(log_data_i['context'], tau=pi_e_params[2])
                    pi_e_distributions_i[policy_name] = action_dist.reshape((action_dist.shape[0],
                                                                             action_dist.shape[1], 1))

            pi_e_dist_train_list.append(pi_e_distributions_i)
            log_data_train_list.append(log_data_i)
            pi_e_dist_test_list.append(pi_e_distributions_i)
            log_data_test_list.append(log_data_i)

        return pi_e_dist_train_list, pi_e_dist_test_list, log_data_train_list, log_data_test_list

    def evaluate_policies_single_task(self, log_data_train, pi_e_dist_train, i_task, log_data_test, pi_e_dist_test,
                                      n_bootstrap=100):
        """
        For given batch data and evaluation policies, we estimate estimator performance with bootstrap and select
        best policy

        Args:
            log_data_train (dict): batch bandit feedback for train (estimator selection)
            pi_e_dist_train (dict): dictionary of evaluation policies,
                keys -> name of policy (str). values -> array of pi_e for train data.
            i_task:
            log_data_test (dict): batch bandit feedback for test (ope and policy selection)
            pi_e_dist_test (dict): dictionary of evaluation policies.
                keys -> name of policy (str). values -> array of pi_e for test data.
            n_bootstrap (int, optional): Defaults to 100. The number of bootstrap sampling in estimator selection.
            If None, we use original data only once.
        Returns:
            pd.DataFrame, dict: dataframe with columns=[policy_name, estimator_name, estimated_policy_value, rank],
            and estimator selection result for each evaluation policy
        """

        # key:policy_name, value:df (estimator_name, mean mse, rank, estimated_policy_value_for_test_data)
        estimator_selection_result = {}
        policy_performance = pd.DataFrame(columns=['policy_name', 'estimator_name', 'estimated_policy_value'])

        for policy_name in pi_e_dist_train.keys():
            start_time = time.time()
            # Estimator Selection
            best_estimator_name, summarized_estimator_selection_result = self.estimator_selection(
                pi_e_dist_train, log_data_train, i_task, n_bootstrap, policy_name
            )
            time_est_sel = time.time() - start_time
            # Off-Policy Evaluation
            estimated_policy_value = self.ope(pi_e_dist_test, log_data_test, i_task, policy_name)
            ope_time = time.time() - time_est_sel

            self.timing = pd.concat([self.timing, pd.DataFrame({'iteration': [i_task],
                                                                'method': [self.policy_selection_name],
                                                                'policy_name': [policy_name],
                                                                'time_est_sel': [time_est_sel],
                                                                'time_ope': [ope_time]})], ignore_index=True)

            policy_performance = pd.concat([policy_performance, pd.DataFrame.from_dict({
                'policy_name': [policy_name],
                'estimator_name': [best_estimator_name],
                'estimated_policy_value': [estimated_policy_value[best_estimator_name]]})], ignore_index=True)

            estimated_policy_value = pd.DataFrame({
                'estimator_name': estimated_policy_value.keys(),
                'estimated_policy_value_for_test_data': estimated_policy_value.values()})
            estimator_selection_result[policy_name] = pd.merge(summarized_estimator_selection_result,
                                                               estimated_policy_value, how='left', on='estimator_name')

        # add rank
        policy_rank = policy_performance.rank(method='min', ascending=False)[['estimated_policy_value']].rename(
            columns={'estimated_policy_value': 'rank'})
        policy_performance = pd.merge(policy_performance, policy_rank, left_index=True, right_index=True)
        time_path = os.path.join(self.log_dir, self.policy_selection_name + '_detailed_timing.csv')
        if not os.path.exists(time_path):
            self.timing.to_csv(time_path)

        return policy_performance, estimator_selection_result

    def ope(self, pi_e_dist_test, log_data_test, i_task, policy_name):
        start_time = time.time()

        # Estimated Policy Value
        est_policy_values_path = self.get_partial_res_file_path_root() + policy_name + 'val_out' + str(i_task)
        if os.path.exists(est_policy_values_path) and self.save:
            estimated_policy_value = pickle.load(open(est_policy_values_path, 'rb'))
        else:
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

                q_model_instance = self.get_q_model_instance(log_data_test['n_rounds'], q_model)
                len_list = 1 if log_data_test['position'] is None else int(log_data_test['position'].max() + 1)

                # estimate policy value
                regression_model = RegressionModelStratified(
                    n_actions=log_data_test['n_actions'],
                    action_context=log_data_test['action_context'],
                    base_model=q_model_instance,
                    len_list=len_list,
                    fitting_method='normal',
                    stratify=self.stratify
                )
                estimated_rewards_by_reg_model = regression_model.fit_predict(
                    context=log_data_test["context"],
                    action=log_data_test["action"],
                    reward=log_data_test["reward"],
                    pscore=log_data_test['pscore'],
                    position=log_data_test['position'],
                    action_dist=log_data_test['pi_b'],
                    n_folds=3,  # use 3-fold cross-fitting
                    random_state=self.random_state
                )
                ope = OffPolicyEvaluation(bandit_feedback=log_data_test, ope_estimators=renamed_ope_estimators)
                estimated_policy_value.update(ope.estimate_policy_values(
                    action_dist=pi_e_dist_test[policy_name],
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
                ))
            if self.save:
                pickle.dump(estimated_policy_value, open(est_policy_values_path, 'wb'))

        return estimated_policy_value

    def estimator_selection(self, pi_e_dist_train, log_data_train, i_task, n_bootstrap, policy_name):
        # Estimator Selection
        per_policy_es_path = self.get_partial_res_file_path_root() + policy_name + 'es_out' + str(i_task)

        if os.path.exists(per_policy_es_path) and self.save:
            estimator_selection = pickle.load(open(per_policy_es_path, 'rb'))
        else:
            estimator_selection = self.do_estimator_selection(pi_e_dist_train, log_data_train, n_bootstrap, policy_name,
                                                              i_task)
            estimator_selection.save_mem_optimized(per_policy_es_path)

        summarized_estimator_selection_result = estimator_selection.get_summarized_results()[
            ['estimator_name', 'mean ' + self.estimator_selection_metrics, 'rank']]

        best_estimator, best_q_model = estimator_selection.get_best_estimator(np.random.RandomState(i_task))
        best_estimator_name = best_estimator.estimator_name
        if 'IPW' not in best_estimator.estimator_name:
            best_estimator_name = best_estimator.estimator_name + '_qmodel_' + best_q_model.__name__

        return best_estimator_name, summarized_estimator_selection_result

    @abstractmethod
    def do_estimator_selection(self, pi_e_dist_train, log_data_train, n_bootstrap, policy_name, i_task):
        """
        Perform estimator selection on D^(pre) dataset, to avoid potential bias

        Args:
            pi_e_dist_train: action distribution by the evaluation policy
            log_data_train: logging data
            n_bootstrap: number of bootstrap sampling iterations
            policy_name: name identifier of the evaluation policy
            i_task: iteration index of the outer loop
        """
        pass

    def get_partial_res_file_path_root(self):
        return self.log_dir + '/partial/' + self.policy_selection_name + '_'


def parallelizable_evaluate_policies_single_task(ps_obj: BasePolicySelection, n_task, n_bootstrap_inference,
                                                 log_data_train, pi_e_dist_train, log_data_test, pi_e_dist_test):
    """
    Utility function used only to parallelize jobs easily
    """
    policy_performance, estimator_selection_result = ps_obj.evaluate_policies_single_task(
        log_data_train=log_data_train, pi_e_dist_train=pi_e_dist_train, i_task=n_task, log_data_test=log_data_test,
        pi_e_dist_test=pi_e_dist_test, n_bootstrap=n_bootstrap_inference
    )
    policy_performance['outer_iteration'] = n_task
    for p_name, e_selection_df in estimator_selection_result.items():
        e_selection_df['outer_iteration'] = n_task
        estimator_selection_result[p_name] = e_selection_df

    return policy_performance, estimator_selection_result
