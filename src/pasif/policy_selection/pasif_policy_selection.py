# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import os
import sys

from common.policy_selection.base_policy_selection import BasePolicySelection

if os.path.dirname(__file__) == '':
    sys.path.append('../../src')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from pasif.estimator_selection.pasif_estimator_selection import PASIFEstimatorSelection


class PASIFPolicySelection(BasePolicySelection):
    """
    Policy-Adaptive policy Selection via Importance Fitting (PAS-IF) method
    """

    def __init__(self, ope_estimators, q_models, stratify, estimator_selection_metrics='mse', data_type='synthetic',
                 random_state=None, log_dir='./'):
        super().__init__(ope_estimators, q_models, stratify, estimator_selection_metrics, data_type, random_state,
                         log_dir, save=True)
        self.pasif_params = None
        self.policy_selection_name = 'pasif'

    def set_pasif_params(self, params):
        """
        Set params for pasif

        Args:
            params (dict): keys: name of policy (str). values: dict (key: param name, value: value of param)
                           ex. { 'beta_1.0':{'k':0.1, 'regularization_weight':0.1, 'batch_size':2000, 'n_epochs':10000,
                           'optimizer':optim.SGD, 'lr':lr }  }
        """
        self.pasif_params = params

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
        pasif = PASIFEstimatorSelection(
            ope_estimators=self.ope_estimators,
            q_models=self.q_models,
            metrics=self.estimator_selection_metrics,
            data_type='real',
            random_state=self.random_state,
            i_task=i_task,
            stratify=self.stratify,
            partial_res_file_name_root=self.get_partial_res_file_path_root() + policy_name + '_',
        )
        pasif.set_real_data(log_data=log_data_train, pi_e_dist=pi_e_dist_train[policy_name])
        pasif.set_pasif_params(
            k=self.pasif_params[policy_name]['k'],
            regularization_weight=self.pasif_params[policy_name]['regularization_weight'],
            batch_size=self.pasif_params[policy_name]['batch_size'],
            n_epochs=self.pasif_params[policy_name]['n_epochs'],
            optimizer=self.pasif_params[policy_name]['optimizer'],
            lr=self.pasif_params[policy_name]['lr']
        )
        pasif.evaluate_estimators(n_bootstrap=n_bootstrap, n_jobs=self.inner_n_jobs)
        return pasif
