import os.path
import pickle
import random
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from obp.utils import check_array
from scipy import stats

from common.data.bootstrap_batch_feedback import sample_bootstrap_batch_bandit_feedback
from common.data.counterfactual_pi_b import get_counterfactual_action_distribution
from common.estimator_selection.abstract_estimator_selection import AbstractEstimatorSelection
from common.regression_model_stratified import RegressionModelStratified



class BaseEstimatorSelection(AbstractEstimatorSelection):
    """
    Base class for policy selection methods
    """



    def __init__(self, ope_estimators, q_models, metrics='mse', data_type='synthetic', random_state=None, i_task=0,
                 partial_res_file_name_root='./base', stratify=True, save=False):
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
        self.n_jobs_fit = 1
        self.stratify = stratify
        assert metrics == 'mse' or metrics == 'mean relative-ee', 'metrics must be mse or mean relative-ee'
        assert data_type == 'synthetic' or data_type == 'real', 'data_type must be synthetic or real'

        self.backend = 'processes'
        self.metrics = metrics
        self.data_type = data_type
        self.partial_res_file_name_root = partial_res_file_name_root
        self.i_task = i_task

        self.n_bootstrap = None
        self.n_jobs = None
        self.synth_pi_e_params = None
        self.synth_n_rounds = None
        self.synth_dataset = None
        self.pi_e_dist = None
        self.log_data = None
        self.save = save



    def set_real_data(self, log_data, pi_e_dist):
        """
        Set real-world data (logged bandit feedback)

        Args:
            log_data (dict): batch bandit feedback
            pi_e_dist (np.array): action distribution for batch bandit feedback by evaluation policy
        """
        self.log_data = log_data
        self.pi_e_dist = pi_e_dist



    def set_synthetic_data(self, synth_dataset, synth_n_rounds, synth_pi_e_params):
        """
        Set synthetic data

        Args:
            synth_dataset (obp.dataset.SyntheticBanditDataset): synthetic data generator
            synth_n_rounds (int): sample size of batch data
            synth_pi_e_params (tuple): evaluation policy.
                          ex. ('beta', 1.0) (using beta to specify evaluation policy)
                          ex. ('function', pi(a|x), tau) (Give any function as evaluation policy. To get action_dist,
                          we use predict_proba(tau=tau))
        """
        assert type(synth_pi_e_params) == tuple, 'type of pi_e must be tuple'
        assert (synth_pi_e_params[0] == 'beta') or (
                synth_pi_e_params[0] == 'function'), 'pi_e[0] must be beta or function'
        assert synth_dataset.random_state is None, 'set random state (int) in dataset'

        self.synth_dataset = synth_dataset
        self.synth_n_rounds = synth_n_rounds
        self.synth_pi_e_params = synth_pi_e_params



    @abstractmethod
    def evaluate_estimators_single_bootstrap_iter(self, log_data, pi_e_dist):
        """
        For given batch data, we estimate estimator performance

        Args:
            log_data (dict): batch bandit feedback
            pi_e_dist (np.array): action distribution by evaluation policy

        Returns:
            dict: key:estimator name, value:metric of estimator performance
        """
        pass



    def evaluate_estimators(self, n_bootstrap=1, n_jobs=None):
        """
        For set data, we evaluate ope estimators with bootstrap for several times

        Args:
            n_bootstrap (int): The number of bootstrap sampling in ope estimator selection. If None, we use
            original data only once.
            n_jobs (int): number of concurrent jobs in loop
        """
        self.n_jobs = n_jobs
        self.n_bootstrap = n_bootstrap

        if self.data_type == 'synthetic':
            if self.synth_pi_e_params[0] == 'beta':
                self.pi_e_dist = get_counterfactual_action_distribution(dataset=self.synth_dataset,
                                                                        cf_beta=self.synth_pi_e_params[1],
                                                                        n_rounds=self.synth_n_rounds)
                self.log_data = self.synth_dataset.obtain_batch_bandit_feedback(n_rounds=self.synth_n_rounds)
            elif self.synth_pi_e_params[0] == 'function':
                self.log_data = self.synth_dataset.obtain_batch_bandit_feedback(n_rounds=self.synth_n_rounds)
                self.pi_e_dist = self.synth_pi_e_params[1].predict_proba(self.log_data['context'],
                                                                         tau=self.synth_pi_e_params[2])
                self.pi_e_dist = self.pi_e_dist.reshape(
                    (self.pi_e_dist.shape[0], self.pi_e_dist.shape[1], 1))

        self._hyperparameters_tuning()

        if self.n_bootstrap is None or self.n_bootstrap <= 1:
            estimator_performance = self.evaluate_estimators_single_bootstrap_iter(log_data=self.log_data,
                                                                                   pi_e_dist=self.pi_e_dist)
            results = [estimator_performance]
        else:
            print('[outer iteration {}] <-----------------  BOOTSTRAP  ----------------->'.format(self.i_task))
            parallel = Parallel(n_jobs=self.n_jobs, verbose=True, prefer=self.backend)
            results = parallel(delayed(parallelizable_evaluate_estimators)
                               (self, self.log_data, self.pi_e_dist, i_bootstrap)
                               for i_bootstrap in range(self.n_bootstrap))

        self.summarize_bootstrap_results(n_bootstrap, results)



    def summarize_bootstrap_results(self, n_bootstrap, results):
        mean_estimator_performance_dict = {}
        for idx, estimator_performance in enumerate(results):
            if idx == 0:
                mean_estimator_performance_dict.update(estimator_performance)
            else:
                for ope_name in mean_estimator_performance_dict.keys():
                    mean_estimator_performance_dict[ope_name] += estimator_performance[ope_name]
        for ope_name in mean_estimator_performance_dict.keys():
            aggr_perf_estimator = mean_estimator_performance_dict[ope_name]
            if n_bootstrap >= 1:
                aggr_perf_estimator /= n_bootstrap
            mean_estimator_performance_dict[ope_name] = aggr_perf_estimator
        self.all_result = pd.DataFrame({
            'estimator_name': mean_estimator_performance_dict.keys(),
            self.metrics: mean_estimator_performance_dict.values()
        })
        estimator_rank = self.all_result.rank(method='min')[[self.metrics]].rename(columns={self.metrics: 'rank'})
        self.all_result = pd.merge(self.all_result, estimator_rank, left_index=True, right_index=True)
        self.all_result['outer_iteration'] = 0
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



    def get_single_bootstrap_performance(self, mean_performance_df):
        '''
        Get estimators performance for one policy i.e., for one outer iteration.
        '''
        estimator_rank = mean_performance_df.rank(method='min')[[self.metrics]].rename(columns={self.metrics: 'rank'})
        mean_performance_df = pd.merge(mean_performance_df, estimator_rank, left_index=True, right_index=True)
        all_result = mean_performance_df
        all_result['outer_iteration'] = self.i_task
        all_result = all_result[['outer_iteration', 'estimator_name', self.metrics, 'rank']]
        summarized_result = pd.DataFrame(columns=['estimator_name', 'mean ' + self.metrics, 'stdev', '95%CI(upper)',
                                                  '95%CI(lower)'])
        for estimator_name in all_result['estimator_name'].unique():
            summarized_result_ = [estimator_name,
                                  all_result[self.metrics][all_result['estimator_name'] == estimator_name].mean()]
            if all_result['outer_iteration'].max() > 0:
                summarized_result_.append(
                    all_result[self.metrics][all_result['estimator_name'] == estimator_name].std())
                t_dist = stats.t(loc=summarized_result_[1],
                                 scale=stats.sem(all_result[self.metrics][
                                                     all_result['estimator_name'] == estimator_name]),
                                 df=len(all_result[self.metrics][all_result['estimator_name'] == estimator_name]) - 1)
                bottom, up = t_dist.interval(alpha=0.95)
                summarized_result_.append(up)
                summarized_result_.append(bottom)
            else:
                summarized_result_.append(None)
                summarized_result_.append(None)
                summarized_result_.append(None)
            summarized_result = pd.concat([summarized_result, pd.DataFrame.from_dict({
                'estimator_name': [summarized_result_[0]],
                'mean ' + self.metrics: [summarized_result_[1]],
                'stdev': [summarized_result_[2]],
                '95%CI(upper)': [summarized_result_[3]],
                '95%CI(lower)': [summarized_result_[4]]
            })], ignore_index=True)
        estimator_rank = summarized_result.rank(method='min')[['mean ' + self.metrics]].rename(
            columns={'mean ' + self.metrics: 'rank'})
        summarized_result = pd.merge(summarized_result, estimator_rank, left_index=True, right_index=True)
        estimator_performance = summarized_result
        estimator_performance = estimator_performance.set_index('estimator_name').to_dict(orient='dict')[
            'mean ' + self.metrics]
        return estimator_performance



    @abstractmethod
    def _hyperparameters_tuning(self):
        """
        Method used to perform hyperparameter tuning
        """
        pass



    def save_mem_optimized(self, per_policy_es_path):
        log_data = self.log_data
        pi_e_dist = self.pi_e_dist

        self.log_data = None
        self.pi_e_dist = None

        pickle.dump(self, open(per_policy_es_path, 'wb'))

        self.log_data = log_data
        self.pi_e_dist = pi_e_dist



    def get_policy_value_estimate(self, ope_estimator, q_model, log_data, pi_e):
        estimated_rwd = self.get_estimated_rewards(log_data, q_model)
        return ope_estimator.estimate_policy_value(reward=log_data['reward'],
                                                   action=log_data['action'],
                                                   action_dist=pi_e,
                                                   estimated_rewards_by_reg_model=estimated_rwd,
                                                   pscore=log_data['pscore'],
                                                   position=log_data['position'],
                                                   estimated_pscore=None)



    def get_policy_value_estimate_given_rwd(self, ope_estimator, log_data, pi_e, estimated_rwd):
        return ope_estimator.estimate_policy_value(reward=log_data['reward'],
                                                   action=log_data['action'],
                                                   action_dist=pi_e,
                                                   estimated_rewards_by_reg_model=estimated_rwd,
                                                   pscore=log_data['pscore'],
                                                   position=log_data['position'],
                                                   estimated_pscore=None)



    def get_estimator_mean_var_given_rwd(self, ope_estimator, log_data, pi_e_dist, estimated_rwd, tune=True):
        round_rwards = estimate_round_rewards(estimated_rwd, log_data, ope_estimator, pi_e_dist, tune)
        return np.mean(round_rwards), np.var(round_rwards) / len(log_data['reward'])



    def get_estimator_var(self, ope_estimator, q_model, log_data, pi_e_dist, tune=True):
        estimated_rwd = self.get_estimated_rewards(log_data, q_model)
        round_rwards = estimate_round_rewards(estimated_rwd, log_data, ope_estimator, pi_e_dist, tune)
        return np.var(round_rwards) / len(log_data['reward'])



    def get_estimated_rewards(self, log_data, q_model):
        if q_model is None:
            return None
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
        return estimated_rewards_by_reg_model



def parallelizable_evaluate_estimators(es_obj: BaseEstimatorSelection, log_data, pi_e_dist, i_bootstrap):
    """
    Utility function used only to parallelize jobs easily
    """
    nn_output_is_nan = True
    additional_i = int(0)
    estimator_performance = None

    file_path = es_obj.partial_res_file_name_root + 'perf_in' + str(i_bootstrap) + '_out' + str(es_obj.i_task)
    if os.path.exists(file_path) and es_obj.save:
        estimator_performance = pickle.load(open(file_path, 'rb'))

    else:
        while nn_output_is_nan and additional_i < 3:
            bootstrapped_data, bootstrapped_dist = sample_bootstrap_batch_bandit_feedback(
                batch_bandit_feedback=log_data,
                action_dist=pi_e_dist,
                sample_size_ratio=1.0,
                random_state=es_obj.random_state + i_bootstrap + (additional_i * es_obj.n_bootstrap)
            )
            estimator_performance = es_obj.evaluate_estimators_single_bootstrap_iter(log_data=bootstrapped_data,
                                                                                     pi_e_dist=bootstrapped_dist)
            if estimator_performance is None:
                additional_i += 1
                print("[outer iteration {}] Additional bootstrap iteration {}".format(es_obj.i_task, additional_i))
            else:
                nn_output_is_nan = False
        if es_obj.save:
            pickle.dump(estimator_performance, open(file_path, 'wb'))
    return estimator_performance



def estimate_round_rewards(estimated_rwd, log_data, ope_estimator, pi_e_dist, tune=True):
    estimator = ope_estimator
    if not hasattr(ope_estimator, '_estimate_round_rewards'):
        if tune:
            check_array(array=log_data['pscore'], name="pscore", expected_dim=1)
            pscore_ = log_data['pscore']
            if log_data['position'] is None:
                log_data['position'] = np.zeros(pi_e_dist.shape[0], dtype=int)
            # tune hyperparameter if necessary
            if not hasattr(ope_estimator, "best_hyperparam"):
                if ope_estimator.tuning_method == "mse":
                    ope_estimator.best_hyperparam = ope_estimator._tune_hyperparam_with_mse(
                        reward=log_data['reward'],
                        action=log_data['action'],
                        pscore=pscore_,
                        action_dist=pi_e_dist,
                        estimated_rewards_by_reg_model=estimated_rwd,
                        position=log_data['position'],
                    )
                elif ope_estimator.tuning_method == "slope":
                    ope_estimator.best_hyperparam = ope_estimator._tune_hyperparam_with_slope(
                        reward=log_data['reward'],
                        action=log_data['action'],
                        pscore=pscore_,
                        action_dist=pi_e_dist,
                        estimated_rewards_by_reg_model=estimated_rwd,
                        position=log_data['position'],
                    )

            estimator = ope_estimator.base_ope_estimator(lambda_=ope_estimator.best_hyperparam,
                                                         use_estimated_pscore=ope_estimator.use_estimated_pscore)
        else:
            estimator = ope_estimator.base_ope_estimator()

    return estimator._estimate_round_rewards(reward=log_data['reward'],
                                             action=log_data['action'],
                                             pscore=log_data['pscore'],
                                             action_dist=pi_e_dist,
                                             estimated_rewards_by_reg_model=estimated_rwd,
                                             position=log_data['position'],
                                             estimated_pscore=None)
