# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
import os
import pickle
import random
import sys
import time

import torch.optim as optim
from joblib import Parallel, delayed

from common.estimator_selection.base_estimator_selection import BaseEstimatorSelection

if os.path.dirname(__file__) == '':
    sys.path.append('../..')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from pasif.estimator_selection.conventional_estimator_selection import ConventionalEstimatorSelection
from pasif.estimator_selection.data_split_pasif import DataSplittingByPasif


class PASIFEstimatorSelection(BaseEstimatorSelection):
    """
    Policy-Adaptive estimator Selection via Importance Fitting (PAS-IF) method
    """

    def __init__(self, ope_estimators, q_models, stratify, metrics='mse', data_type='synthetic', random_state=None, i_task=0,
                 partial_res_file_name_root='./pasif'):
        super().__init__(ope_estimators, q_models, metrics, data_type, random_state, i_task,
                         partial_res_file_name_root, save=True)
        self.stratify = stratify
        self.pasif_optimizer = None
        self.pasif_n_epochs = None
        self.pasif_original_regularization_weight = None
        self.pasif_batch_size = None
        self.pasif_lr = None
        self.pasif_k = None
        self.pass_tuning = None
        self.pass_alpha = None
        self.pass_k = None

    def set_pasif_params(self, k=0.1, regularization_weight=1.0, batch_size=None, n_epochs=100, optimizer=optim.SGD,
                         lr=0.01):
        """
        Set params for PASIF

        Args:
            k (float, optional): Defaults to 0.1. Expected sample ratio of subsumpled data
            regularization_weight (float, optional): Defaults to 1.0. Coefficient of regularization in loss.
            batch_size (int, optional): Defaults to None. Batch size in training. If None, we set sample size as 
            batch_size.
            n_epochs (int, optional): Defaults to 100. Number of epochs.
            optimizer (torch.optim, optional): Defaults to optim.SGD. Optimizer of nn.
            lr (float, optional): Defaults to 0.01. Learning rate.
        """
        self.pasif_k = k
        self.pasif_original_regularization_weight = regularization_weight
        self.pasif_batch_size = batch_size
        self.pasif_n_epochs = n_epochs
        self.pasif_optimizer = optimizer
        self.pasif_lr = lr

    def _split_data(self, log_data, pi_e_dist, process_id):
        """
        Split data into 2 quasi log data

        Args:
            log_data (dict): batch bandit feedback
            pi_e_dist (np.array): action dist by evaluation policy
            process_id (int): inner iteration id

        Returns:
            dict, dict, np.array, np.array: batch1, counterfactual action dist1, batch2, counterfactual action dist2
        """
        data_split = DataSplittingByPasif(
            batch_bandit_feedback=log_data,
            action_dist_by_pi_e=pi_e_dist,
            random_state=self.random_state
        )
        data_split.set_params(
            k=self.pasif_k,
            regularization_weight=self.pasif_regularization_weight,
            batch_size=self.pasif_batch_size,
            n_epochs=self.pasif_n_epochs,
            optimizer=self.pasif_optimizer,
            lr=self.pasif_lr
        )

        start_time = time.time()
        device = data_split.train_importance_fitting(process_id)
        print('[Task ', str(self.i_task), ', Iteration ', str(process_id), ']: Importance fitting time:', time.time() - start_time)
        # print('\t\t\tMemory Allocated After:', torch.cuda.memory_allocated(device))
        # print('\t\t\tMemory Reserved After:', torch.cuda.memory_reserved(device))

        batch_feedback_1, cf_action_dist_1, batch_feedback_2, cf_action_dist_2 = data_split.get_split_data()

        self.pasif_final_loss = data_split.final_loss
        self.pasif_final_loss_d = data_split.final_loss_d
        self.pasif_final_loss_r = data_split.final_loss_r
        self.pasif_split_info_dict = data_split.split_info_dict

        return batch_feedback_1, cf_action_dist_1, batch_feedback_2, cf_action_dist_2

    def evaluate_estimators_single_bootstrap_iter(self, log_data, pi_e_dist):
        """
        For given batch data, we estimate estimator performance

        Args:
            log_data (dict): batch bandit feedback
            pi_e_dist (np.array): action distribution by evaluation policy

        Returns:
            dict: key:estimator name, value:metric of estimator performance
        """
        # split data
        batch_bandit_feedback_1, action_dist_1_by_2, batch_bandit_feedback_2, action_dist_2_by_1 = \
            self._split_data(log_data=log_data, pi_e_dist=pi_e_dist, process_id=self.i_task)

        if batch_bandit_feedback_1 is None:
            # This is needed for grid search of weight
            return None
        else:
            # using split data as two log data, we use conventional estimator selection method
            conventional_method = ConventionalEstimatorSelection(
                ope_estimators=self.ope_estimators,
                q_models=self.q_models,
                metrics=self.metrics,
                data_type='real',
                random_state=self.random_state,
                stratify=self.stratify
            )
            conventional_method.set_real_data(
                batch_bandit_feedback_1=batch_bandit_feedback_1,
                action_dist_1_by_2=action_dist_1_by_2,
                batch_bandit_feedback_2=batch_bandit_feedback_2,
                action_dist_2_by_1=action_dist_2_by_1,
                evaluation_data='1'
            )
            conventional_method.evaluate_estimators(
                n_inner_bootstrap=None,
                n_outer_bootstrap=None,
                outer_repeat_type='bootstrap',
                ground_truth_method='on_policy'
            )

            estimator_performance = conventional_method.get_summarized_results()
            estimator_performance = estimator_performance.set_index('estimator_name').to_dict(orient='dict')[
                'mean ' + self.metrics]

        return estimator_performance

    def _hyperparameters_tuning(self):
        """
        Method used to perform hyperparameter tuning on the regularization coefficient
        """
        file_path = self.partial_res_file_name_root + 'weight_grid_search_result' + str(self.i_task)
        candidate_regularization_weight = [1e-1, 1e0, 1e1, 1e2, 1e3]

        if self.pasif_original_regularization_weight in [-999, -998, -997]:
            print('[outer iteration {}] <----------------- GRID SEARCH ----------------->'.format(self.i_task))
            weight_grid_search_result = {}  # key:weight, value:metrics to choose weight
            if os.path.exists(file_path) and self.save:
                weight_grid_search_result = pickle.load(open(file_path, 'rb'))
            else:
                parallel = Parallel(n_jobs=self.n_jobs, verbose=True, prefer=self.backend)
                results = parallel(delayed(parallelizable_grid_search)(self, temp_regularization_weight)
                                   for temp_regularization_weight in candidate_regularization_weight)

                for weight, res in zip(candidate_regularization_weight, results):
                    weight_grid_search_result[weight] = res
                if self.save:
                    pickle.dump(weight_grid_search_result, open(file_path, 'wb'))

            if self.pasif_original_regularization_weight in [-999, -998]:
                # select weight with minimum metrics
                self.pasif_regularization_weight = min(weight_grid_search_result, key=weight_grid_search_result.get)

            elif self.pasif_original_regularization_weight == -997:
                # select weight with minimal loss D and satisfying the condition for k
                print("Weight grid search results:", weight_grid_search_result)
                self.pasif_regularization_weight = None
                minimum_loss = None
                for temp_regularization_weight, tuple_of_metrics in weight_grid_search_result.items():
                    if not (tuple_of_metrics[0] is None):
                        if (tuple_of_metrics[0] > self.pasif_k - 0.02) and (tuple_of_metrics[0] < self.pasif_k + 0.02):
                            if minimum_loss is None:
                                minimum_loss = tuple_of_metrics[1]
                                self.pasif_regularization_weight = temp_regularization_weight
                            else:
                                if tuple_of_metrics[1] < minimum_loss:
                                    minimum_loss = tuple_of_metrics[1]
                                    self.pasif_regularization_weight = temp_regularization_weight

            # check results of grid search
            if self.pasif_original_regularization_weight == -997:
                if self.pasif_regularization_weight is None:
                    self.pasif_regularization_weight = random.choice(candidate_regularization_weight)
                    print('There were no candidates of regularization_weight that met the conditions for k')
                    print('Chosen a random regularization weight', self.pasif_regularization_weight)
                    path = self.partial_res_file_name_root + 'NO_REG_WEIGHT_FOUND' + str(self.i_task) + '.pickle'
                    if self.save:
                        pickle.dump(self, open(path, 'wb'))
        else:
            self.pasif_regularization_weight = self.pasif_original_regularization_weight



def parallelizable_grid_search(es: PASIFEstimatorSelection, temp_regularization_weight):
    """
    Utility function used only to parallelize grid search jobs easily
    """
    es.pasif_regularization_weight = temp_regularization_weight
    es.evaluate_estimators_single_bootstrap_iter(es.log_data, es.pi_e_dist)

    res = None
    if es.pasif_original_regularization_weight == -999:
        # metrics: abs diff between loss_d and loss_r
        res = abs(es.pasif_final_loss_d - es.pasif_final_loss_r)

    elif es.pasif_original_regularization_weight == -998:
        # metrics: abs diff between loss_d and regularization_weight * loss_r
        res = abs(es.pasif_final_loss_d - temp_regularization_weight * es.pasif_final_loss_r)

    elif es.pasif_original_regularization_weight == -997:
        # metrics: tuple of mean marginal_p and final loss D
        if not (es.pasif_split_info_dict['marginal_p'] is None):
            res = (es.pasif_split_info_dict['marginal_p'].mean(), es.pasif_final_loss_d)
        else:
            res = (None, es.pasif_final_loss_d)

    return res
