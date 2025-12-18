import os
import pickle
import random
import time
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
from numpy.random import RandomState
from torch import optim

from black_box.policy_selection.auto_ope_policy_selection import AutoOPEPolicySelection
from common.evaluation.evaluation_metric import calculate_relative_regret_p, calculate_rank_correlation_coefficient_p, \
    calculate_relative_regret_e, calculate_rank_correlation_coefficient_e, calculate_mse_e
from pasif.policy_selection.ocv_policy_selection import OCVPolicySelection
from pasif.policy_selection.pasif_policy_selection import PASIFPolicySelection
from pasif.policy_selection.slope_policy_selection import SLOPEPolicySelection



class BaseEvaluation(metaclass=ABCMeta):
    def __init__(self, ope_estimators, q_models, log_dir_path, test_ratio=0.5, n_data_generation=10, random_state=None,
                 estimator_selection_metrics='mse', pi_e={}, outer_n_jobs=-1, inner_n_jobs=None, n_bootstrap=10,
                 outer_n_jobs_gt=1, stratify=True):
        self.black_box_es_res = None
        self.black_box_ps_res = None
        self.ope_estimators = ope_estimators
        self.ope_estimators_names = None
        self.q_models = q_models
        self.pi_e = pi_e
        self.filter_estimators = False
        self.data_type = None

        self.n_data_generation = n_data_generation
        self.test_ratio = test_ratio
        self.estimator_selection_metrics = estimator_selection_metrics
        self.outer_n_jobs_gt = outer_n_jobs_gt
        self.outer_n_jobs = outer_n_jobs
        self.inner_n_jobs = inner_n_jobs
        self.n_bootstrap = n_bootstrap
        self.stratify = stratify
        self.log_dir_path_load = log_dir_path
        self.log_dir_path_save = log_dir_path
        os.makedirs(self.log_dir_path_load + '/partial/', exist_ok=True)

        self.ocv_valid_estimators, self.ocv_valid_q_models, self.ocv_K, self.ocv_train_ratio, self.ocv_one_stderr_rule = None, None, None, None, None
        self.const_est_names_list = None
        self.metadata_avg, self.metadata_n_boot, self.metadata_n_points, self.metadata_rwd_type = None, None, None, None
        self.model_name, self.custom_folder, self.opes_type, self.output_type, self.metric_opt, self.use_embeddings = None, None, None, None, None, None
        self.c_evaluation_data = None
        self.p_method_params = None
        self.p_method_name = None
        self.slope_estimators_supported = None

        self.mean_es_evaluation_metrics_list = []
        self.es_eval_list = []
        self.mean_ps_evaluation_metrics_list = []
        self.ps_eval_list = []

        self.estimator_selection_gt, self.policy_selection_gt = {}, pd.DataFrame(columns=['policy_name', 'policy_value'])
        self.random_state = random.randint(1, 10000000) if random_state is None else random_state
        self.rng = RandomState(self.random_state)



    def set_ground_truth(self, load_gt, n_sampling=1):
        """
        set ground truth of estimator/policy selection

        Args:
            load_gt:
        """
        print('Set ground truth {}'.format(time.gmtime()))
        self.estimator_selection_gt = {}
        self.policy_selection_gt = pd.DataFrame(columns=['policy_name', 'policy_value'])

        if load_gt:
            self.estimator_selection_gt = pickle.load(open(self.log_dir_path_load + '/estimator_selection_gt.pickle', 'rb'))
            self.policy_selection_gt = pickle.load(open(self.log_dir_path_load + '/policy_selection_gt.pickle', 'rb'))
        else:
            for policy_name, policy in self.pi_e.items():
                # ground truth of estimator performance
                c_es, policy_value = self.get_gt_policy(policy, policy_name)

                estimator_performance = c_es.get_summarized_results()
                estimator_performance = estimator_performance[
                    ['estimator_name', 'mean ' + self.estimator_selection_metrics, 'rank']].rename(
                    columns={'mean ' + self.estimator_selection_metrics: self.estimator_selection_metrics})
                self.estimator_selection_gt[policy_name] = estimator_performance

                # ground truth of evaluation policy
                self.policy_selection_gt = pd.concat([self.policy_selection_gt, pd.DataFrame.from_dict(
                    {'policy_name': [policy_name], 'policy_value': [policy_value]})], ignore_index=True)

            policy_rank = self.policy_selection_gt.rank(method='min', ascending=False)[['policy_value']].rename(
                columns={'policy_value': 'rank'})
            res = pd.merge(self.policy_selection_gt, policy_rank, left_index=True, right_index=True, copy=False)
            self.policy_selection_gt = res

        pickle.dump(self.estimator_selection_gt, open(self.log_dir_path_save + '/estimator_selection_gt.pickle', 'wb'))
        pickle.dump(self.policy_selection_gt, open(self.log_dir_path_save + '/policy_selection_gt.pickle', 'wb'))
        self.save_mem_optimized()



    def set_conventional_method_params(self, evaluation_data='partial_random'):
        """
        set params for conventional estimator/policy selection

        Args:
            evaluation_data (str, optional): Defaults to 'partial_random'. Must be '1' or '2' or 'random' or
            'partial_random'.
                                             Which data (behavior policy) to consider as evaluation policy.
                                             'partial_random' means that we use fixed data as evalu policy in bootstrap,
                                             but no fixed in outer loop in estimator selection.
        """
        self.c_evaluation_data = evaluation_data



    def set_slope_params(self, slope_estimators_supported):
        """
        set params for conventional estimator/policy selection

        Args:
            evaluation_data (str, optional): Defaults to 'partial_random'. Must be '1' or '2' or 'random' or
            'partial_random'.
                                             Which data (behavior policy) to consider as evaluation policy.
                                             'partial_random' means that we use fixed data as evalu policy in bootstrap,
                                             but no fixed in outer loop in estimator selection.
        """
        self.slope_estimators_supported = slope_estimators_supported



    def set_pasif_method_params(self, method_name, method_params):
        """
        set params for PAS-IF estimator/policy selection

        Args:
            method_name (str)): name of data splitting method. Must be pass or pasif.
            method_params (dict): params for data splitting.
                                  if you use pass, this is dict of "key:name of param, value:value of param".
                                  ex (pass). {'k':2.0, 'alpha':1.0, 'tuning':False}
                                  if you use pasif, this is dict of "key:name of policy, value:dict (key:name of param,
                                  value:value of param)".
                                  ex (pasif). { 'beta_1.0':{'k':0.1, 'regularization_weight':0.1, 'batch_size':2000,
                                  'n_epochs':10000, 'optimizer':optim.SGD, 'lr':lr }  }
        """
        self.p_method_name = method_name
        self.p_method_params = method_params



    def set_ocv_params(self, valid_estimators, valid_q_models, K, train_ratio, one_stderr_rule):
        self.ocv_valid_estimators = valid_estimators
        self.ocv_valid_q_models = valid_q_models
        self.ocv_K = K
        self.ocv_train_ratio = train_ratio
        self.ocv_one_stderr_rule = one_stderr_rule



    def set_bb_params(self, model_name, opes_type, output_type, metric_opt, use_embeddings, metadata_rwd_type,
                      metadata_n_points, metadata_n_boot, metadata_avg, custom_folder):
        self.model_name = model_name
        self.opes_type = opes_type
        self.output_type = output_type
        self.metric_opt = metric_opt
        self.use_embeddings = use_embeddings
        self.metadata_n_points = metadata_n_points
        self.metadata_n_boot = metadata_n_boot
        self.metadata_avg = metadata_avg
        self.metadata_rwd_type = metadata_rwd_type
        self.custom_folder = custom_folder



    def set_const_params(self, est_names_list):
        self.const_est_names_list = est_names_list



    def policy_estimator_selection_evaluation(self, policy_selection, res_load, prefix):
        es_res_file = prefix + '_est_selection_res.pickle'
        ps_res_file = prefix + '_policy_selection_res.pickle'

        if not res_load:
            policy_selection.evaluate_policies(n_inference_bootstrap=self.n_bootstrap, n_task=self.n_data_generation,
                                               test_ratio=self.test_ratio, outer_n_jobs=self.outer_n_jobs,
                                               inner_n_jobs=self.inner_n_jobs)
            est_selection_res = policy_selection.get_all_estimator_selection_results()
            policy_selection_res = policy_selection.get_all_results()
        else:
            #already_filtered = (self.filter_estimators and os.path.exists(os.path.join(self.log_dir_path_save, ps_res_file)) and
            #                    os.path.exists(os.path.join(self.log_dir_path_save, es_res_file)))
            #if already_filtered:
            #    es_res_file = os.path.join(self.log_dir_path_save, es_res_file)
            #    ps_res_file = os.path.join(self.log_dir_path_save, ps_res_file)
            #else:
            #    es_res_file = os.path.join(self.log_dir_path_load, es_res_file)
            #    ps_res_file = os.path.join(self.log_dir_path_load, ps_res_file)
            es_res_file = os.path.join(self.log_dir_path_load, es_res_file)
            ps_res_file = os.path.join(self.log_dir_path_load, ps_res_file)

            est_selection_res = pickle.load(open(es_res_file, 'rb'))
            policy_selection_res = pickle.load(open(ps_res_file, 'rb'))
            #if not already_filtered:
            #    est_selection_res, policy_selection_res = self.filter_results_by_estimators(est_selection_res, policy_selection_res)

        pickle.dump(est_selection_res, open(os.path.join(self.log_dir_path_save, es_res_file), 'wb'))
        pickle.dump(policy_selection_res, open(os.path.join(self.log_dir_path_save, ps_res_file), 'wb'))

        tot_estimator_selection_eval = self.estimator_selection_evaluation(est_selection_res)
        tot_policy_selection_eval = self.policy_selection_evaluation(policy_selection_res)
        pickle.dump(tot_estimator_selection_eval, open(self.log_dir_path_save + '/' + prefix +
                                                       '_tot_estimator_selection_eval.pickle', 'wb'))
        pickle.dump(tot_policy_selection_eval, open(self.log_dir_path_save + '/' + prefix +
                                                    '_tot_policy_selection_eval.pickle', 'wb'))
        self.save_mem_optimized()
        self.append_mean_es_evaluation_metrics(tot_estimator_selection_eval, prefix)
        self.es_eval_list.append((prefix, tot_estimator_selection_eval))
        self.append_mean_ps_evaluation_metrics(tot_policy_selection_eval, prefix)
        self.ps_eval_list.append((prefix, tot_policy_selection_eval))

        return est_selection_res, policy_selection_res



    def policy_selection_evaluation(self, policy_selection_res):
        tot_policy_selection_eval = pd.DataFrame(
            columns=['outer_iteration', 'relative_regret', 'rank_correlation_coefficient'])

        for outer_loop in range(self.n_data_generation):
            predict_data = policy_selection_res[policy_selection_res['outer_iteration'] == outer_loop]
            true_data = self.policy_selection_gt
            relative_regret_p = calculate_relative_regret_p(true_data=true_data, estimated_data=predict_data,
                                                            random=self.rng)
            rank_cc_p = calculate_rank_correlation_coefficient_p(true_data=true_data, estimated_data=predict_data)
            tot_policy_selection_eval = pd.concat([tot_policy_selection_eval, pd.DataFrame.from_dict({
                'outer_iteration': [outer_loop],
                'relative_regret': [relative_regret_p],
                'rank_correlation_coefficient': [rank_cc_p]
            })], ignore_index=True)
        return tot_policy_selection_eval



    def estimator_selection_evaluation(self, estimator_selection_res):
        for policy_name, result_df in estimator_selection_res.items():
            estimator_selection_res[policy_name] = result_df.rename(
                columns={'mean ' + self.estimator_selection_metrics: 'estimated ' + self.estimator_selection_metrics})

        tot_estimator_selection_eval = {}  # key:policy_name, value:dataframe
        for policy_name, estimator_selection_result in estimator_selection_res.items():
            estimator_selection_eval = pd.DataFrame(
                columns=['outer_iteration', 'relative_regret', 'rank_correlation_coefficient', 'mse'])
            for outer_loop in range(self.n_data_generation):
                predict_data = estimator_selection_result[estimator_selection_result['outer_iteration'] == outer_loop]
                true_data = self.estimator_selection_gt[policy_name]
                relative_regret_e, random_est = calculate_relative_regret_e(
                    true_data=true_data, estimated_data=predict_data, random=self.rng,
                    estimator_selection_metrics=self.estimator_selection_metrics
                )
                rank_cc_e = calculate_rank_correlation_coefficient_e(true_data=true_data, estimated_data=predict_data)
                mse = calculate_mse_e(
                    true_data=true_data, estimated_data=predict_data, random_est=random_est,
                    estimator_selection_metrics=self.estimator_selection_metrics
                )
                estimator_selection_eval = pd.concat([estimator_selection_eval, pd.DataFrame.from_dict({
                    'outer_iteration': [outer_loop],
                    'relative_regret': [relative_regret_e],
                    'rank_correlation_coefficient': [rank_cc_e],
                    'mse': [mse]
                })], ignore_index=True)
            tot_estimator_selection_eval[policy_name] = estimator_selection_eval
        return tot_estimator_selection_eval



    def evaluate_constant(self):
        """
        evaluate PAS-IF estimator/policy selection
        """
        const_es_res = deepcopy(self.black_box_es_res)
        n_estimators = len(const_es_res[list(const_es_res.keys())[0]]['estimator_name'].unique())

        for const_est_name in self.const_est_names_list:
            const_es_res = deepcopy(self.black_box_es_res)
            for policy_name, es_res in const_es_res.items():
                es_res.loc[es_res['estimator_name'] != const_est_name, 'estimated mse'] = np.inf
                es_res.loc[es_res['estimator_name'] != const_est_name, 'rank'] = np.nan
                es_res.loc[es_res['estimator_name'] == const_est_name, 'rank'] = 1
                if const_est_name not in es_res['estimator_name'] and 'opera' in const_est_name:  # This is for OPERA
                    for i in range(es_res['outer_iteration'].max() + 1):
                        new_row = pd.DataFrame([[const_est_name, 0.0, 1, 0.0, i]],
                                               columns=es_res.columns,
                                               index=[n_estimators])
                        const_es_res[policy_name] = pd.concat([const_es_res[policy_name], new_row], ignore_index=True)

            const_ps_res = deepcopy(self.black_box_ps_res)  # TODO: implement correct results for constant policy selection

            pickle.dump(const_es_res, open(self.log_dir_path_save + '/constant_{}_est_selection_res.pickle'.format(const_est_name), 'wb'))
            pickle.dump(const_ps_res, open(self.log_dir_path_save + '/constant_{}_policy_selection_res.pickle'.format(const_est_name), 'wb'))
            self.policy_estimator_selection_evaluation(None, True, 'constant_{}'.format(const_est_name))



    def evaluate_random(self):
        """
        evaluate PAS-IF estimator/policy selection
        """
        # Empirical Average
        random_es_res = deepcopy(self.black_box_es_res)
        estimator_names = random_es_res[list(random_es_res.keys())[0]]['estimator_name'].unique()
        random_ps_res = pd.DataFrame(index=[], columns=self.black_box_ps_res.columns)

        for policy_name, results_df in random_es_res.items():
            results_df.loc[:, 'rank'] = 1  # so that all estimators can be sampled in policy_estimator_selection_evaluation

        random_ps_res = self.from_es_to_ps_results(random_es_res, random_ps_res)

        pickle.dump(random_es_res, open(self.log_dir_path_save + '/sampled_random_est_selection_res.pickle', 'wb'))
        pickle.dump(random_ps_res, open(self.log_dir_path_save + '/sampled_random_policy_selection_res.pickle', 'wb'))
        self.policy_estimator_selection_evaluation(None, True, 'sampled_random')

        # Expectation
        random_tot_estimator_selection_eval = {}
        policy_selection_res = pd.DataFrame(index=[], columns=['outer_iteration', 'policy_name',
                                                                           'estimator_name', 'estimated_policy_value',
                                                                           'rank'])

        for policy_name, estimator_selection_result in self.estimator_selection_gt.items():
            estimator_selection_eval = pd.DataFrame(
                columns=['outer_iteration', 'relative_regret', 'rank_correlation_coefficient', 'mse'])
            for outer_loop in range(self.n_data_generation):
                relative_regret_e_mean = 0
                true_data = self.estimator_selection_gt[policy_name]
                true_best_estimator_name_list = true_data['estimator_name'][true_data['rank'] == 1].values
                true_best_estimator_name = true_best_estimator_name_list[0]
                true_estimator_performance = true_data['mse'][true_data['estimator_name'] == true_best_estimator_name].values[0]
                for estimator in estimator_names:
                    relative_regret_e = (true_data['mse'][true_data['estimator_name'] == estimator].values[0] / true_estimator_performance) - 1.0
                    relative_regret_e_mean += relative_regret_e
                relative_regret_e_mean /= true_data.shape[0]
                rank_cc_e = 0.0
                mse = true_data['mse'].mean()

                estimator_selection_eval = pd.concat([estimator_selection_eval, pd.DataFrame.from_dict({
                    'outer_iteration': [outer_loop],
                    'relative_regret': [relative_regret_e_mean],
                    'rank_correlation_coefficient': [rank_cc_e],
                    'mse': [mse]
                })], ignore_index=True)
            random_tot_estimator_selection_eval[policy_name] = estimator_selection_eval

            policy_selection_res = pd.concat([policy_selection_res, pd.DataFrame.from_dict(
                {
                    'outer_iteration': range(self.n_data_generation),
                    'policy_name': [policy_name] * self.n_data_generation,
                    'estimator_name': ["TODO"] * self.n_data_generation,
                    'estimated_policy_value': ["TODO"] * self.n_data_generation,
                    'rank': ["TODO"] * self.n_data_generation
                    })], ignore_index=True)
        pickle.dump(random_tot_estimator_selection_eval, open(self.log_dir_path_save + '/' + 'exact_random' +
                                                              '_tot_estimator_selection_eval.pickle', 'wb'))

        self.append_mean_es_evaluation_metrics(random_tot_estimator_selection_eval, 'exact_random')
        self.es_eval_list.append(('exact_random', random_tot_estimator_selection_eval))
        #self.append_mean_ps_evaluation_metrics(random_tot_policy_selection_eval, 'exact_random')
        #self.ps_eval_list.append(('exact_random', random_tot_policy_selection_eval))



    def from_es_to_ps_results(self, random_es_res, random_ps_res):
        rnd = RandomState(self.random_state)
        for policy_name, results_df in random_es_res.items():
            # Policy Selection
            best_policy_estimators = results_df[results_df['rank'] == 1].rename(columns={
                'estimated_policy_value_for_test_data': 'estimated_policy_value'})
            best_policy_estimators['policy_name'] = policy_name

            for out_it in range(best_policy_estimators['outer_iteration'].max() + 1):
                iter_mask = best_policy_estimators['outer_iteration'] == out_it
                iter_data = best_policy_estimators[iter_mask]
                if iter_data.shape[0] > 1:
                    sampled_estimator = pd.DataFrame(iter_data.iloc[rnd.choice(list(iter_data.index)), :]).T
                else:
                    sampled_estimator = iter_data
                random_ps_res = pd.concat([random_ps_res, sampled_estimator])

        random_ps_res.drop(['estimated mse'], inplace=True, axis=1)
        df_tmp_correct_rank = pd.DataFrame(index=[], columns=random_ps_res.columns)
        for out_it in range(random_ps_res['outer_iteration'].max() + 1):
            outer_iter_df = random_ps_res[random_ps_res['outer_iteration'] == out_it].copy()
            rank = outer_iter_df.rank(method='min', ascending=False)[['estimated_policy_value']].rename(
                columns={'estimated_policy_value': 'rank'})
            outer_iter_df['rank'] = rank
            df_tmp_correct_rank = pd.concat([df_tmp_correct_rank, outer_iter_df])
        random_ps_res = df_tmp_correct_rank
        return random_ps_res



    def append_mean_es_evaluation_metrics(self, es_eval, es_method_name):
        mean_metrics, std_metrics = {}, {}
        if es_eval is not None:
            for policy_name, estimator_selection_result in es_eval.items():
                mean_metrics[policy_name] = {
                    'relative_regret': estimator_selection_result['relative_regret'].mean(),
                    'rank_correlation_coefficient': estimator_selection_result['rank_correlation_coefficient'].mean(),
                    'mse': estimator_selection_result['mse'].mean()
                }
                std_metrics[policy_name] = {
                    'relative_regret': estimator_selection_result['relative_regret'].std(),
                    'rank_correlation_coefficient': estimator_selection_result['rank_correlation_coefficient'].std(),
                    'mse': estimator_selection_result['mse'].std()
                }
        self.mean_es_evaluation_metrics_list.append((es_method_name, mean_metrics, std_metrics))



    def append_mean_ps_evaluation_metrics(self, ps_eval, ps_method_name):
        mean_results, std_results = {}, {}
        rank_key, regr_key = 'rank_correlation_coefficient', 'relative_regret'
        if ps_eval is not None:
            mean_results[regr_key] = ps_eval[regr_key].mean()
            mean_results[rank_key] = ps_eval[rank_key].mean()
            std_results[regr_key] = ps_eval[regr_key].std()
            std_results[rank_key] = ps_eval[rank_key].std()
            self.mean_ps_evaluation_metrics_list.append((ps_method_name, mean_results, std_results))



    def get_mean_evaluation_results_of_estimator_selection(self):
        """
        Get mean evaluation results of estimator selection methods

        Returns:
            dict, dict, dict, dict, dict, dict: results for methods. key:policy name.
            value:dict of mean or std evaluation metrics.
        """
        return self.mean_es_evaluation_metrics_list



    def get_all_evaluation_results_of_estimator_selection(self):
        """
        Get all evaluation results of estimator selection methods

        Returns:
            dict, dict, dict: results for methods. key:policy name. value:dataframe of
            evaluation results.
        """
        return self.es_eval_list



    def get_mean_evaluation_results_of_policy_selection(self):
        """
        Get mean evaluation results of policy selection methods

        Returns:
            dict, dict, dict, dict, dict, dict: results for methods. key:name of
            evaluation metrics. value:mean or std of result.
            c_mean_results, p_mean_results, bb_mean_results, c_std_results, p_std_results, bb_std_results
        """
        return self.mean_ps_evaluation_metrics_list



    def get_all_evaluation_results_of_policy_selection(self):
        """
        Get all evaluation results of policy selection methods

        Returns:
            pd.DataFrame, pd.DataFrame, pd.DataFrame: results for methods.
        """
        return self.ps_eval_list



    @abstractmethod
    def save_mem_optimized(self):
        pass



    def filter_gt_by_estimators(self, results_dict: Dict[str, pd.DataFrame]):
        if not self.filter_estimators:
            return

        if self.ope_estimators_names is None:
            self.ope_estimators_names = []
            for i, q_model in enumerate(self.q_models):
                for ope_estimator in deepcopy(self.ope_estimators):
                    if (not "IPW" in ope_estimator.estimator_name) or i == 0:
                        if 'IPW' not in ope_estimator.estimator_name:
                            ope_estimator.estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                        self.ope_estimators_names.append(ope_estimator.estimator_name)

        for policy_name, results_df in results_dict.items():
            results_df = results_df[results_df['estimator_name'].isin(self.ope_estimators_names)]
            rank = results_df.rank(method='min', ascending=True)[['mse']].rename(columns={'mse': 'rank'})
            results_df['rank'] = rank
            results_dict[policy_name] = results_df

        return results_dict



    def filter_results_by_estimators(self, es_results_dict: Dict[str, pd.DataFrame], ps_results: pd.DataFrame):
        if not self.filter_estimators:
            return

        if self.ope_estimators_names is None:
            self.ope_estimators_names = []
            for i, q_model in enumerate(self.q_models):
                for ope_estimator in deepcopy(self.ope_estimators):
                    if (not "IPW" in ope_estimator.estimator_name) or i == 0:
                        if 'IPW' not in ope_estimator.estimator_name:
                            ope_estimator.estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                        self.ope_estimators_names.append(ope_estimator.estimator_name)

        ps_results_new = pd.DataFrame(index=[], columns=ps_results.columns)

        # Estimator Selection
        for policy_name, results_df in es_results_dict.items():
            results_df = results_df[results_df['estimator_name'].isin(self.ope_estimators_names)]

            df_tmp_correct_rank = pd.DataFrame(index=[], columns=results_df.columns)
            for out_it in range(results_df['outer_iteration'].max() + 1):
                outer_iter_df = results_df[results_df['outer_iteration'] == out_it].copy()
                rank = outer_iter_df.rank(method='min', ascending=True)[['mean mse']].rename(columns={'mean mse': 'rank'})
                outer_iter_df['rank'] = rank
                df_tmp_correct_rank = pd.concat([df_tmp_correct_rank, outer_iter_df])
            results_df = df_tmp_correct_rank

            es_results_dict[policy_name] = results_df

            # Policy Selection
            ps_results_new = pd.concat([ps_results_new, results_df[results_df['rank'] == 1].rename(columns={
                'estimated_policy_value_for_test_data': 'estimated_policy_value'})])
            ps_results_new['policy_name'] = policy_name

        ps_results_new.drop(['mean mse'], inplace=True, axis=1)
        df_tmp_correct_rank = pd.DataFrame(index=[], columns=ps_results_new.columns)
        for out_it in range(ps_results_new['outer_iteration'].max() + 1):
            outer_iter_df = ps_results_new[ps_results_new['outer_iteration'] == out_it].copy()
            rank = outer_iter_df.rank(method='min', ascending=False)[['estimated_policy_value']].rename(
                columns={'estimated_policy_value': 'rank'})
            outer_iter_df['rank'] = rank
            df_tmp_correct_rank = pd.concat([df_tmp_correct_rank, outer_iter_df])
        ps_results_new = df_tmp_correct_rank

        return es_results_dict, ps_results_new



    def evaluate_conventional_selection_method(self, load):
        return



    def evaluate_pasif(self, load):
        """
        evaluate PAS-IF estimator/policy selection
        """
        print('Evaluation of PAS-IF selection method {}'.format(time.gmtime()))
        pasif_ps = None
        if not load:
            pasif_ps = PASIFPolicySelection(ope_estimators=self.ope_estimators, q_models=self.q_models,
                                            stratify=self.stratify,
                                            estimator_selection_metrics=self.estimator_selection_metrics,
                                            data_type=self.data_type, random_state=self.random_state,
                                            log_dir=self.log_dir_path_load)
            self.set_data(pasif_ps)
            if self.p_method_name == 'pasif':
                pasif_ps.set_pasif_params(params=self.p_method_params)

        self.policy_estimator_selection_evaluation(pasif_ps, load, 'pasif')



    def evaluate_ocv(self, load):
        """
        evaluate black-box estimator/policy selection
        """
        print('Evaluation of Off-Policy Cross-Validation selection method {}'.format(time.gmtime()))
        for ocv_valid_estimator, ocv_valid_q_model in zip(self.ocv_valid_estimators, self.ocv_valid_q_models):
            ocv_ps = None
            if not load:
                ocv_ps = OCVPolicySelection(ope_estimators=self.ope_estimators, q_models=self.q_models,
                                            estimator_selection_metrics=self.estimator_selection_metrics,
                                            data_type=self.data_type, random_state=self.random_state,
                                            log_dir=self.log_dir_path_load, stratify=self.stratify)
                self.set_data(ocv_ps)
                ocv_ps.set_ocv_params(valid_estimator=ocv_valid_estimator, valid_q_model=ocv_valid_q_model,
                                      K=self.ocv_K, train_ratio=self.ocv_train_ratio,
                                      one_stderr_rule=self.ocv_one_stderr_rule)

            name = 'ocv_' + ocv_valid_estimator.estimator_name
            if ocv_valid_q_model is not None:
                name += "_" + ocv_valid_q_model.__name__

            self.policy_estimator_selection_evaluation(ocv_ps, load, name)



    def evaluate_slope(self, load):
        """
        evaluate black-box estimator/policy selection
        """
        print('Evaluation of SLOPE selection method {}'.format(time.gmtime()))
        slope_ps = None
        if not load:
            slope_ps = SLOPEPolicySelection(ope_estimators=self.ope_estimators, q_models=self.q_models,
                                            estimator_selection_metrics=self.estimator_selection_metrics,
                                            data_type=self.data_type, random_state=self.random_state,
                                            log_dir=self.log_dir_path_load, stratify=self.stratify)
            slope_ps.set_slope_params(self.slope_estimators_supported)
            self.set_data(slope_ps)

        self.policy_estimator_selection_evaluation(slope_ps, load, 'slope')



    def evaluate_bb_selection_method(self, load):
        """
        evaluate black-box estimator/policy selection
        """
        print('Evaluation of AutoOPE selection method {}'.format(time.gmtime()))
        black_box_ps = None
        if not load:
            black_box_ps = AutoOPEPolicySelection(ope_estimators=self.ope_estimators, q_models=self.q_models,
                                                  estimator_selection_metrics=self.estimator_selection_metrics,
                                                  data_type=self.data_type, random_state=self.random_state,
                                                  log_dir=self.log_dir_path_load, stratify=self.stratify)
            self.set_data(black_box_ps)
            black_box_ps.set_bb_params(self.model_name, self.opes_type, self.output_type, self.metric_opt,
                                       self.use_embeddings, self.metadata_rwd_type, self.metadata_n_points,
                                       self.metadata_n_boot, self.metadata_avg, self.custom_folder)

        self.black_box_es_res, self.black_box_ps_res = self.policy_estimator_selection_evaluation(black_box_ps, load,
                                                                                                  'black_box')



    @abstractmethod
    def set_data(self, ps):
        pass



    def set_pasif_method_params_wrapper(self, arguments, policies_names=None):
        pasif_optimizer_dict = {0: optim.SGD, 1: optim.Adam}
        method_param_dict = {}
        for policy_name_for_key, pasif_k, pasif_rw, pasif_bs, pasif_n_eopchs, pasif_opt, pasif_lr in \
                zip(policies_names, arguments.pasif_k, arguments.pasif_regularization_weight,
                    arguments.pasif_batch_size,
                    arguments.pasif_n_epochs, arguments.pasif_optimizer, arguments.pasif_lr):
            method_param_dict[policy_name_for_key] = {
                'k': pasif_k,
                'regularization_weight': pasif_rw,
                'batch_size': pasif_bs,
                'n_epochs': pasif_n_eopchs,
                'optimizer': pasif_optimizer_dict[pasif_opt],
                'lr': pasif_lr
            }
        self.set_pasif_method_params(method_name='pasif', method_params=method_param_dict)



    @abstractmethod
    def get_gt_policy(self, policy, policy_name):
        pass