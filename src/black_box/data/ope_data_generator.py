import glob
import os
import pickle
from time import time
from typing import List, Optional, Dict, Callable, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import Generator
from obp.dataset.reward_type import RewardType
from obp.ope import BaseOffPolicyEstimator
from scipy.stats import randint, rv_discrete, rv_continuous
from sklearn.base import BaseEstimator

from black_box.common.constants import META_DATASET_FOLDER_PATH, DATA_FOLDER_NAME
from black_box.common.utils import check_array_elem, add_param_to_dict, mapping, get_metadataset_name
from black_box.data.synthetic_ope_data import SyntheticOffPolicyContextBanditData



class DataGenerator:
    INDEX_COL_NAME = 'index'
    MAX_SEED = 2 ** 32 - 1
    FILE_EXTENSION = '.pickle'



    def __init__(self, random_state: int, op_estimators: List[BaseOffPolicyEstimator], n_jobs: int = 1,
                 binary_rwd_models: List[BaseEstimator.__class__] = [],
                 continuous_rwd_models: List[BaseEstimator.__class__] = []):
        self.ope_performance_generated = None
        self.tmp_dir = None
        self.reward_types = None
        self.initialized = False
        self.op_estimators = op_estimators
        self.binary_rwd_models = binary_rwd_models
        self.continuous_rwd_models = continuous_rwd_models
        self.random_state = random_state
        self.rng = np.random.RandomState(seed=random_state)
        self.gen_idx = 0
        self.n_jobs = n_jobs
        self.synthetic_generations = np.ndarray(shape=(1, 1))
        self.random_state_arr = None

        # Rewards
        self.binary_id = 0
        self.continuous_id = 1
        self.bin_perc = None
        self.bin_cont_mask = None
        self.continuous_count = None
        self.binary_count = None
        self.binary_rwd_funcs = None
        self.binary_rwd_funcs_prob = None
        self.continuous_rwd_funcs = None
        self.continuous_rwd_funcs_prob = None
        self.rwd_std_distr = None
        self.kwargs_rwd_var = None
        self.reward_types_arr = None
        self.reward_functions_arr = None
        self.reward_std_arr = None

        # Target Policies
        self.op_betas_distr = None
        self.op_kwargs_betas = None
        self.op_policies_funcs = None
        self.op_policies_funcs_prob = None
        self.op_beta_arr = None
        self.target_policies_arr = None

        # Behavioural Policies
        self.multi_beta_prob = None
        self.betas_distr = None
        self.kwargs_betas = None
        self.policies_funcs = None
        self.policies_funcs_prob = None
        self.behavior_policy_function_arr = None
        self.beta_arr = None

        # Context
        self.kwargs_contexts_dims = None
        self.distribution_contexts_dims = None
        self.context_dims_arr = None

        # N Actions
        self.distribution_nactions = None
        self.kwargs_nactions = None
        self.n_actions_arr = None
        self.action_context_arr = None

        # Deficient Actions
        self.distribution_n_def_actions = None
        self.no_def_action_prob = 1.0
        self.upper_bound_param_name_def_act = None
        self.kwargs_n_def_actions = None
        self.n_deficient_actions_arr = None

        # OP Deficient Actions
        self.op_upper_bound_param_name_def_act = None
        self.op_no_def_action_prob = 1.0
        self.op_distribution_n_def_actions = None
        self.op_kwargs_n_def_actions = None
        self.op_n_deficient_actions_arr = None

        # N Rounds
        self.distribution_nrounds = None
        self.kwargs_nrounds = None
        self.n_rounds_arr = None

        self.dataset_name_arr = None
        self.n_points = None
        self.n_bootstrap = None
        self.gt_points = None
        self.gt_subsamples = None



    def generate_tuples(self, n_points: int, n_bootstrap: int = 1, gt_points: int = 100000, gt_subsamples: int = 10,
                        force_regeneration: bool = False, batch_size: int = 'auto'):
        """
        generate n_points points for each estimator in op_estimator
        :param n_points:
        :param n_bootstrap:
        :param gt_points:
        :param gt_subsamples:
        :param force_regeneration:
        :param batch_size:
        :return:
        """

        self._prepare_distributions(gt_points, gt_subsamples, n_bootstrap, n_points)
        all_rwd_models = self.binary_rwd_models + self.continuous_rwd_models
        embeddings = SyntheticOffPolicyContextBanditData.get_estimator_features(self.op_estimators, all_rwd_models)

        if force_regeneration:
            self.ope_performance_generated = [False] * self.n_points

        # Data Generation
        start_time = time()
        batch_size = batch_size if batch_size > 0 else 'auto'
        parallel = Parallel(n_jobs=self.n_jobs, verbose=10, batch_size=batch_size)
        gen_data = parallel(delayed(_generation)(self, idx) for idx in range(self.n_points))
        self._save_generated_data(embeddings, gen_data, start_time)



    def _prepare_distributions(self, gt_points, gt_subsamples, n_bootstrap, n_points):
        # Initialize distributions
        assert self.op_estimators is not None, "No estimators provided."
        self.n_bootstrap = n_bootstrap
        self.n_points = n_points
        self.gt_points = gt_points
        self.gt_subsamples = gt_subsamples
        self.gen_idx += 1
        self._set_binary_reward_probability()
        self._set_nrounds_distribution()
        self._set_nactions_distribution()
        self._set_contexts_dims_distribution()
        self._set_rewards_distribution()
        self._set_behaviour_policies_distribution()
        self._set_target_policies_distribution()
        self._set_random_states()
        self.dataset_name_arr = []
        for ctx_idx in range(self.n_points):
            self.dataset_name_arr.append('synthetic_dataset_' + str(ctx_idx))
        if not self.initialized:
            self.initialized = True

        # Setup log dirs
        self.tmp_dir = self._get_meta_dataset_folder()[0] + '/tmp'
        os.makedirs(self.tmp_dir, exist_ok=True)
        pickle.dump(self, open(self.tmp_dir + '/generator' + self.FILE_EXTENSION, 'wb'))
        tmpdir_d = self.tmp_dir + '/tuples'
        os.makedirs(tmpdir_d, exist_ok=True)

        # Check if all performance tmp files are present
        self.ope_performance_generated = []
        for idx in range(self.n_points):
            file_name_noavg_se = tmpdir_d + '/se' + str(idx) + self.FILE_EXTENSION
            file_name_noavg_rel = tmpdir_d + '/rel_ee' + str(idx) + self.FILE_EXTENSION
            file_name_noavg_err = tmpdir_d + '/errors' + str(idx) + self.FILE_EXTENSION
            self.ope_performance_generated.append(os.path.exists(file_name_noavg_se) and
                                                  os.path.exists(file_name_noavg_rel) and
                                                  os.path.exists(file_name_noavg_err))
        true_count = self.ope_performance_generated.count(True)
        false_count = self.ope_performance_generated.count(False)
        print("How many generated: {}, How many to generate: {}".format(true_count, false_count))



    def _save_generated_data(self, embeddings, gen_data, start_time):
        b_data_contexts_df = [tuple_[0] for tuple_ in gen_data]
        b_op_relative_errors_df = [tuple_[1] for tuple_ in gen_data]
        b_op_squared_errors_df = [tuple_[2] for tuple_ in gen_data]
        avg_data_contexts_df = [tuple_[3] for tuple_ in gen_data]
        avg_op_relative_errors_df = [tuple_[4] for tuple_ in gen_data]
        avg_op_squared_errors_df = [tuple_[5] for tuple_ in gen_data]
        errors = [tuple_[6] for tuple_ in gen_data]

        errors = pd.DataFrame(errors)
        res_bootstrapped = (b_data_contexts_df, b_op_relative_errors_df, b_op_squared_errors_df)
        res_avg = (avg_data_contexts_df, avg_op_relative_errors_df, avg_op_squared_errors_df)
        for data_contexts_df, op_relative_errors_df, op_squared_errors_df in (res_bootstrapped, res_avg):
            data_contexts_df = pd.concat(data_contexts_df, axis=0, ignore_index=True)
            op_relative_errors_df = pd.concat(op_relative_errors_df, axis=0, ignore_index=True)
            op_squared_errors_df = pd.concat(op_squared_errors_df, axis=0, ignore_index=True)

            data_contexts_df.set_index(keys=self.INDEX_COL_NAME, inplace=True)
            op_squared_errors_df.set_index(keys=self.INDEX_COL_NAME, inplace=True)
            op_relative_errors_df.set_index(keys=self.INDEX_COL_NAME, inplace=True)

            self._save_csv(file_name="errors", data_to_save=errors, common=True)
            self._save_csv(file_name="se", data_to_save=op_squared_errors_df)
            self._save_csv(file_name="rel_ee", data_to_save=op_relative_errors_df)
            self._save_csv(file_name="ctx", data_to_save=data_contexts_df)
            self._save_csv(file_name="est_embed", data_to_save=embeddings, common=True)
            if start_time:
                self._save_csv(file_name="time", data_to_save=pd.DataFrame.from_dict(
                    {'sec': [time() - start_time]}), common=True)
            self._save_csv(file_name="synthetic_datasets_params", data_to_save=pd.DataFrame.from_dict(
                {"n_rounds": self.n_rounds_arr,
                 "n_actions": self.n_actions_arr,
                 "dim_context": self.context_dims_arr,
                 "reward_type": self.reward_types_arr,
                 "reward_function": [fun.__name__ if fun is not None else None for fun in
                                     self.reward_functions_arr],
                 "reward_std": self.reward_std_arr,
                 "behavior_policy_function": self.behavior_policy_function_arr,
                 "beta": self.beta_arr,
                 "n_deficient_actions": self.n_deficient_actions_arr,
                 "dataset_name": self.dataset_name_arr,
                 "cf_policy_function": [fun.__name__ if fun is not None else None for fun in
                                        self.target_policies_arr],
                 "cf_beta": self.op_beta_arr,
                 "cf_n_deficient_actions": self.op_n_deficient_actions_arr,
                 "random_state": self.random_state_arr,
                 "n_bootstrap": self.n_bootstrap * np.ones(shape=self.n_points)}),
                           common=True)



    def _save_csv(self, file_name: str, data_to_save: pd.DataFrame, common: bool = False):
        meta_dataset_folder, suffix = self._get_meta_dataset_folder()
        if not common and self.n_bootstrap > 1:
            if data_to_save.shape[0] >= self.n_points:
                avg_folder = "noavg" if data_to_save.shape[0] > self.n_points else "avg"
                suffix += "_" + avg_folder
                meta_dataset_folder = os.path.join(meta_dataset_folder, avg_folder)
                meta_dataset_folder = os.path.join(meta_dataset_folder, DATA_FOLDER_NAME)
        os.makedirs(meta_dataset_folder, exist_ok=True)
        data_to_save.to_csv(os.path.join(meta_dataset_folder, file_name + "_" + suffix + ".csv"), index=False)



    def _get_meta_dataset_folder(self):
        meta_dataset_folder = META_DATASET_FOLDER_PATH
        suffix = get_metadataset_name(self.reward_types, self.n_points, self.n_bootstrap)
        meta_dataset_folder = os.path.join(os.path.join(meta_dataset_folder, suffix))
        os.makedirs(meta_dataset_folder, exist_ok=True)
        return meta_dataset_folder, suffix



    def set_nrounds_distribution(self, distribution: rv_discrete, **kwargs) -> None:
        if not self.initialized:
            self.distribution_nrounds = distribution
            self.kwargs_nrounds = kwargs



    def set_nactions_distribution(self, distribution: rv_discrete, **kwargs) -> None:
        if not self.initialized:
            self.distribution_nactions = distribution
            self.kwargs_nactions = kwargs



    def set_n_def_actions_distribution(self, distribution: rv_discrete, no_def_action_prob: float = 0.5,
                                       upper_bound_param_name: str = "", **kwargs) -> None:
        if not self.initialized:
            self.distribution_n_def_actions = distribution
            self.kwargs_n_def_actions = kwargs
            self.no_def_action_prob = no_def_action_prob
            self.upper_bound_param_name_def_act = upper_bound_param_name



    def set_op_n_def_actions_distribution(self, distribution: rv_discrete, no_def_action_prob: float = 0.5,
                                          upper_bound_param_name: str = "", **kwargs) -> None:
        if not self.initialized:
            self.op_distribution_n_def_actions = distribution
            self.op_kwargs_n_def_actions = kwargs
            self.op_no_def_action_prob = no_def_action_prob
            self.op_upper_bound_param_name_def_act = upper_bound_param_name



    def set_contexts_dims_distribution(self, distribution: rv_discrete, **kwargs) -> None:
        if not self.initialized:
            self.distribution_contexts_dims = distribution
            self.kwargs_contexts_dims = kwargs



    def set_rewards_distribution(self,
                                 binary_rwd_funcs: List[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                                 continuous_rwd_funcs: List[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                                 rwd_std_distr: rv_continuous = None,
                                 binary_rwd_funcs_prob: Optional[List[float]] = None,
                                 continuous_rwd_funcs_prob: Optional[List[float]] = None,
                                 **kwargs) -> None:
        if not self.initialized:
            self.binary_rwd_funcs = binary_rwd_funcs
            if not self.binary_rwd_funcs:
                self.binary_rwd_funcs = None
            if binary_rwd_funcs_prob is None and self.binary_rwd_funcs is not None:
                uniform_prob = 1 / len(self.binary_rwd_funcs)
                self.binary_rwd_funcs_prob = [uniform_prob] * len(self.binary_rwd_funcs)
            else:
                self.binary_rwd_funcs_prob = binary_rwd_funcs_prob

            self.continuous_rwd_funcs = continuous_rwd_funcs
            if not self.continuous_rwd_funcs:
                self.continuous_rwd_funcs = None
            if continuous_rwd_funcs_prob is None and self.continuous_rwd_funcs is not None:
                uniform_prob = 1 / len(self.continuous_rwd_funcs)
                self.continuous_rwd_funcs_prob = [uniform_prob] * len(self.continuous_rwd_funcs)
            else:
                self.continuous_rwd_funcs_prob = continuous_rwd_funcs_prob
            self.rwd_std_distr = rwd_std_distr
            self.kwargs_rwd_var = kwargs



    def set_behaviour_policies_distribution(self,
                                            policies_funcs: List[
                                                Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]],
                                            betas_distr: rv_continuous = None,
                                            multi_beta_prob: float = 0.0,
                                            policies_funcs_prob: Optional[List[float]] = None,
                                            **kwargs) -> None:
        if not self.initialized:
            self.policies_funcs = policies_funcs
            if not self.policies_funcs:
                self.policies_funcs = None
            if policies_funcs_prob is None and self.policies_funcs is not None:
                uniform_prob = 1 / len(self.policies_funcs)
                self.policies_funcs_prob = [uniform_prob] * len(self.policies_funcs)
            else:
                self.policies_funcs_prob = policies_funcs_prob
            self.betas_distr = betas_distr
            self.multi_beta_prob = multi_beta_prob
            self.kwargs_betas = kwargs



    def set_target_policies_distribution(self,
                                         policies_funcs: List[Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]],
                                         betas_distr: rv_continuous = None,
                                         policies_funcs_prob: Optional[List[float]] = None,
                                         **kwargs) -> None:
        if not self.initialized:
            self.op_policies_funcs = policies_funcs
            if not self.op_policies_funcs:
                self.op_policies_funcs = None
            if policies_funcs_prob is None and self.op_policies_funcs is not None:
                uniform_prob = 1 / len(self.op_policies_funcs)
                self.op_policies_funcs_prob = [uniform_prob] * len(self.op_policies_funcs)
            else:
                self.op_policies_funcs_prob = policies_funcs_prob
            self.op_betas_distr = betas_distr
            self.op_kwargs_betas = kwargs



    def set_binary_reward_probability(self, bin_perc: float = 0.5) -> None:
        if not self.initialized:
            self.bin_perc = bin_perc
            if self.bin_perc >= 1.:
                self.reward_types = "bin"
            elif self.bin_perc <= 0.:
                self.reward_types = "cont"
            else:
                self.reward_types = "mix"



    def _set_nrounds_distribution(self) -> None:
        if self.distribution_nrounds is not None:
            self.n_rounds_arr = np.array(
                self.distribution_nrounds.rvs(size=self.n_points, **self.kwargs_nrounds, random_state=self.rng),
                dtype=int)



    def _set_nactions_distribution(self) -> None:
        if self.distribution_nactions is not None:
            self.n_actions_arr = np.array(
                self.distribution_nactions.rvs(size=self.n_points, **self.kwargs_nactions, random_state=self.rng),
                dtype=int)
        self.n_deficient_actions_arr = self._set_n_def_action_distribution(self.no_def_action_prob,
                                                                           self.upper_bound_param_name_def_act,
                                                                           self.kwargs_n_def_actions,
                                                                           self.distribution_n_def_actions)
        self.op_n_deficient_actions_arr = self._set_n_def_action_distribution(self.op_no_def_action_prob,
                                                                              self.op_upper_bound_param_name_def_act,
                                                                              self.op_kwargs_n_def_actions,
                                                                              self.op_distribution_n_def_actions)



    def _set_n_def_action_distribution(self,
                                       no_def_action_prob: float = 1.0,
                                       no_def_act_str_par: str = None,
                                       kwargs_n_def_actions: Dict = None,
                                       distribution_n_def_actions: rv_discrete = None) -> np.ndarray:
        no_id = 0
        yes_id = 1
        type_ids = (no_id, yes_id)
        n_deficient_actions_arr = rv_discrete(name='not_deficient_actions', values=(
            type_ids, (no_def_action_prob, 1 - no_def_action_prob))).rvs(
            size=self.n_points, random_state=self.rng)
        if no_def_act_str_par is not None and distribution_n_def_actions is not None:
            def_act_arr_tmp = []
            for n_max_action in self.n_actions_arr:
                def_act_arr_tmp.append(distribution_n_def_actions.rvs(
                    size=1, random_state=self.rng, **add_param_to_dict(no_def_act_str_par, n_max_action - 1,
                                                                       kwargs_n_def_actions)))
            def_act_arr_tmp = np.array(def_act_arr_tmp)
            def_act_arr_tmp = def_act_arr_tmp.reshape(self.n_actions_arr.shape)[n_deficient_actions_arr == yes_id]
            n_deficient_actions_arr[n_deficient_actions_arr == yes_id] = def_act_arr_tmp
        return n_deficient_actions_arr



    def _set_contexts_dims_distribution(self) -> None:
        if self.distribution_contexts_dims is not None:
            self.context_dims_arr = np.array(
                self.distribution_contexts_dims.rvs(size=self.n_points, random_state=self.rng,
                                                    **self.kwargs_contexts_dims),
                dtype=int)



    def _set_rewards_distribution(self) -> None:
        self.reward_functions_arr = np.ndarray(self.bin_cont_mask.shape, dtype=object)
        self.reward_std_arr = np.ndarray(self.bin_cont_mask.shape, dtype=float)

        # Continuous
        if self.continuous_rwd_funcs is not None:
            cont_fun_ids = range(len(self.continuous_rwd_funcs))
            continuous_rwd_functions_arr = rv_discrete(name='cont_reward_function',
                                                       values=(cont_fun_ids, self.continuous_rwd_funcs_prob)).rvs(
                size=self.continuous_count, random_state=self.rng)
            continuous_rwd_functions_arr = mapping(continuous_rwd_functions_arr, cont_fun_ids,
                                                   self.continuous_rwd_funcs)
            self.reward_functions_arr[self.bin_cont_mask == self.continuous_id] = continuous_rwd_functions_arr

            self.reward_std_arr[self.bin_cont_mask == self.continuous_id] = np.array(
                self.rwd_std_distr.rvs(size=self.continuous_count, random_state=self.rng, **self.kwargs_rwd_var),
                dtype=float)

        # Binary
        if self.binary_rwd_funcs is not None:
            bin_fun_ids = range(len(self.binary_rwd_funcs))
            binary_rwd_functions_arr = rv_discrete(name='bin_reward_function',
                                                   values=(bin_fun_ids, self.binary_rwd_funcs_prob)).rvs(
                size=self.binary_count, random_state=self.rng)
            binary_rwd_functions_arr = mapping(binary_rwd_functions_arr, bin_fun_ids, self.binary_rwd_funcs)
            self.reward_functions_arr[self.bin_cont_mask == self.binary_id] = binary_rwd_functions_arr
            self.reward_std_arr[self.bin_cont_mask == self.binary_id] = 1.0



    def _set_behaviour_policies_distribution(self) -> None:
        self.behavior_policy_function_arr, self.beta_arr = self._set_policies_distribution(
            policies_funcs=self.policies_funcs,
            policies_funcs_prob=self.policies_funcs_prob,
            betas_distr=self.betas_distr,
            kwargs_betas=self.kwargs_betas,
            multi_betas_prob=self.multi_beta_prob)



    def _set_target_policies_distribution(self) -> None:
        self.target_policies_arr, self.op_beta_arr = self._set_policies_distribution(
            policies_funcs=self.op_policies_funcs,
            policies_funcs_prob=self.op_policies_funcs_prob,
            betas_distr=self.op_betas_distr,
            kwargs_betas=self.op_kwargs_betas,
            multi_betas_prob=0.)



    def _set_policies_distribution(self,
                                   policies_funcs: List[Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]],
                                   policies_funcs_prob: List[float],
                                   betas_distr: rv_continuous,
                                   kwargs_betas: Dict,
                                   multi_betas_prob: float) -> (np.ndarray, np.ndarray):
        beta_arr = None
        pol_ids = range(len(self.policies_funcs))
        behavior_policy_function_arr = rv_discrete(name='b_policy_distribution',
                                                   values=(pol_ids, policies_funcs_prob)).rvs(
            size=self.n_points, random_state=self.rng)
        behavior_policy_function_arr = mapping(behavior_policy_function_arr, pol_ids, policies_funcs)

        if betas_distr is not None:
            single_beta_points = int((1 - multi_betas_prob) * self.n_points)
            multi_beta_points = self.n_points - single_beta_points
            multi_beta_size = (multi_beta_points, 2)
            single_beta_arr = np.array(betas_distr.rvs(size=single_beta_points, random_state=self.rng, **kwargs_betas),
                                       dtype=float).tolist()
            multi_beta_arr = np.array(betas_distr.rvs(size=multi_beta_size, random_state=self.rng, **kwargs_betas),
                                      dtype=float).tolist()
            beta_arr = single_beta_arr + multi_beta_arr
            self.rng.shuffle(beta_arr)
        return behavior_policy_function_arr, beta_arr



    def _set_random_states(self):
        self.random_state_arr = randint.rvs(size=self.n_points, random_state=self.rng, low=0, high=self.MAX_SEED)



    def _set_binary_reward_probability(self) -> None:
        type_ids = (self.binary_id, self.continuous_id)
        self.bin_cont_mask = rv_discrete(name='reward_type', values=(type_ids, (self.bin_perc, 1 - self.bin_perc))).rvs(
            size=self.n_points, random_state=self.rng)
        self.reward_types_arr = mapping(self.bin_cont_mask, type_ids,
                                        [RewardType.BINARY.value, RewardType.CONTINUOUS.value])

        unique_arr, occurences = np.unique(self.bin_cont_mask, return_counts=True)
        only_bin, only_cont = False, False
        if len(unique_arr) == 1:
            if unique_arr[0] == self.binary_id:
                only_bin = True
            else:
                only_cont = True
        if not only_cont:
            self.binary_count = occurences[self.binary_id]
        if not only_bin:
            self.continuous_count = occurences[self.continuous_id]



def _generation(gen: DataGenerator, idx: int):
    tmpdir_gt = gen.tmp_dir + '/groundtruth'
    os.makedirs(tmpdir_gt, exist_ok=True)
    file_name = tmpdir_gt + '/groundtruth' + str(idx) + gen.FILE_EXTENSION
    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
    else:
        data = SyntheticOffPolicyContextBanditData(
            n_rounds=check_array_elem(gen.n_rounds_arr, idx, dflt=1),
            n_actions=check_array_elem(gen.n_actions_arr, idx, dflt=2),
            dim_context=check_array_elem(gen.context_dims_arr, idx, dflt=1),
            reward_type=check_array_elem(gen.reward_types_arr, idx, dflt=RewardType.BINARY.value),
            reward_function=check_array_elem(gen.reward_functions_arr, idx, dflt=None),
            reward_std=check_array_elem(gen.reward_std_arr, idx, dflt=1.0),
            action_context=check_array_elem(gen.action_context_arr, idx, dflt=None),
            behavior_policy_function=check_array_elem(gen.behavior_policy_function_arr, idx, dflt=None),
            beta=check_array_elem(gen.beta_arr, idx, dflt=1.0),
            n_deficient_actions=check_array_elem(gen.n_deficient_actions_arr, idx, dflt=0),
            dataset_name=check_array_elem(gen.dataset_name_arr, idx, dflt='synthetic_dataset_' + str(idx)),
            cf_policy_function=check_array_elem(gen.target_policies_arr, idx, dflt=None),
            cf_beta=check_array_elem(gen.op_beta_arr, idx, dflt=1.0),
            cf_n_deficient_actions=check_array_elem(gen.op_n_deficient_actions_arr, idx, dflt=0),
            random_state=check_array_elem(gen.random_state_arr, idx, dflt=randint(low=0, high=gen.MAX_SEED)),
            n_bootstrap=gen.n_bootstrap,
            gt_points=gen.gt_points,
            gt_subsamples=gen.gt_subsamples
        )
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)

    tmpdir_d = gen.tmp_dir + '/tuples'
    os.makedirs(tmpdir_d, exist_ok=True)

    found_files = [glob.glob(e) for e in [os.path.join(tmpdir_d, 'features' + str(idx) + gen.FILE_EXTENSION),
                                          os.path.join(tmpdir_d, 'errors' + str(idx) + gen.FILE_EXTENSION),
                                          os.path.join(tmpdir_d, 'rel_ee' + str(idx) + gen.FILE_EXTENSION),
                                          os.path.join(tmpdir_d, 'se' + str(idx) + gen.FILE_EXTENSION)]]
    if [] in found_files:
        found_files.remove([])

    boot_features, boot_se, boot_rel_ee, errors = None, None, None, None

    if len(found_files) < 4:  # no files are generated (or are generated only performance)
        boot_features, boot_se, boot_rel_ee, errors = data.generation(ope_estimators=gen.op_estimators,
                                                                      binary_rwd_models=gen.binary_rwd_models,
                                                                      continuous_rwd_models=gen.continuous_rwd_models,
                                                                      compute_ope_perf=not
                                                                      gen.ope_performance_generated[idx])
        pickle.dump(boot_features, open(tmpdir_d + '/features' + str(idx) + gen.FILE_EXTENSION, 'wb'))
    else:
        boot_features = pickle.load(open(tmpdir_d + '/features' + str(idx) + gen.FILE_EXTENSION, 'rb'))

    if not gen.ope_performance_generated[idx]:
        pickle.dump(boot_se, open(tmpdir_d + '/se' + str(idx) + gen.FILE_EXTENSION, 'wb'))
        pickle.dump(boot_rel_ee, open(tmpdir_d + '/rel_ee' + str(idx) + gen.FILE_EXTENSION, 'wb'))
        pickle.dump(errors, open(tmpdir_d + '/errors' + str(idx) + gen.FILE_EXTENSION, 'wb'))

    else:
        boot_se = pickle.load(open(tmpdir_d + '/se' + str(idx) + gen.FILE_EXTENSION, 'rb'))
        boot_rel_ee = pickle.load(open(tmpdir_d + '/rel_ee' + str(idx) + gen.FILE_EXTENSION, 'rb'))
        errors = pickle.load(open(tmpdir_d + '/errors' + str(idx) + gen.FILE_EXTENSION, 'rb'))

        if not (boot_rel_ee.shape[0] == gen.n_bootstrap and boot_se.shape[0] == gen.n_bootstrap):
            boot_features, boot_se, boot_rel_ee, errors = data.generation(ope_estimators=gen.op_estimators,
                                                                          binary_rwd_models=gen.binary_rwd_models,
                                                                          continuous_rwd_models=gen.continuous_rwd_models,
                                                                          compute_ope_perf=True)


    avg_boot_features, avg_boot_rel_ee, avg_boot_se = data.average_on_bootstrap(boot_features, boot_rel_ee, boot_se)

    bootstrap_idx = range(idx * gen.n_bootstrap, (idx + 1) * gen.n_bootstrap)
    boot_features[gen.INDEX_COL_NAME] = bootstrap_idx
    boot_rel_ee[gen.INDEX_COL_NAME] = bootstrap_idx
    boot_se[gen.INDEX_COL_NAME] = bootstrap_idx
    avg_boot_features[gen.INDEX_COL_NAME] = idx
    avg_boot_rel_ee[gen.INDEX_COL_NAME] = idx
    avg_boot_se[gen.INDEX_COL_NAME] = idx

    return boot_features, boot_rel_ee, boot_se, avg_boot_features, avg_boot_rel_ee, avg_boot_se, errors
