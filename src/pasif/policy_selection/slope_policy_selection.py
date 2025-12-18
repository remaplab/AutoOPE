import os
import time

import numpy as np
import pandas as pd

from common.policy_selection.base_policy_selection import BasePolicySelection
from pasif.estimator_selection.slope_estimator_selection import SLOPEEstimatorSelection



class SLOPEPolicySelection(BasePolicySelection):
    """
     black-box off-policy policy selection method
    """

    def __init__(self, ope_estimators, q_models, stratify, estimator_selection_metrics='mse', data_type='synthetic',
                 random_state=None, log_dir='./'):
        super().__init__(ope_estimators, q_models, stratify, estimator_selection_metrics, data_type, random_state, log_dir)
        self.estimators_supported = None
        self.custom_folder = None
        self.metadata_rwd_type = None
        self.metadata_avg = None
        self.metadata_n_boot = None
        self.metadata_n_points = None
        self.use_embeddings = None
        self.metric_opt = None
        self.output_type = None
        self.opes_type = None
        self.model_name = None
        self.policy_selection_name = 'slope'
        self.backend = 'threads'



    def set_slope_params(self, estimators_supported):
        self.estimators_supported = estimators_supported



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
        # estimator selection
        slope_es = SLOPEEstimatorSelection(
            ope_estimators=self.ope_estimators,
            q_models=self.q_models,
            estimators_supported=self.estimators_supported,
            metrics=self.estimator_selection_metrics,
            data_type='real',
            random_state=self.random_state,
            i_task=i_task,
            partial_res_file_name_root=self.get_partial_res_file_path_root() + policy_name + '_',
            stratify=self.stratify,
        )
        slope_es.set_real_data(log_data=log_data_train, pi_e_dist=pi_e_dist_train[policy_name])
        slope_es.evaluate_estimators(n_bootstrap=n_bootstrap, n_jobs=self.inner_n_jobs)
        return slope_es
