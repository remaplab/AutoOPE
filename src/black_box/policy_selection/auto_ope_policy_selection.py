import os
import time

import numpy as np
import pandas as pd

from black_box.common.utils import opes_factory, get_metadataset_working_dir
from black_box.data.data_load_utils import load_split
from black_box.data.synthetic_ope_data import SyntheticOffPolicyContextBanditData
from black_box.estimator_selection.auto_ope_estimator_selection import AutoOPEEstimatorSelection
from common.policy_selection.base_policy_selection import BasePolicySelection


class AutoOPEPolicySelection(BasePolicySelection):
    """
     black-box off-policy policy selection method
    """

    def __init__(self, ope_estimators, q_models, stratify, estimator_selection_metrics='mse', data_type='synthetic',
                 random_state=None, log_dir='./'):
        super().__init__(ope_estimators, q_models, stratify, estimator_selection_metrics, data_type, random_state, log_dir)
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
        self.policy_selection_name = 'black-box'
        self.backend = 'threads'

    def _prepare_model(self):
        # Initializing a random number generator
        rng = np.random.RandomState(seed=self.random_state)
        working_dir = get_metadataset_working_dir(self.metadata_rwd_type, self.metadata_n_points, self.metadata_n_boot,
                                                  self.metadata_avg)
        _, _, _, _, embed, _ = load_split(test_perc=0.5, train_features_plots=False, rng=rng,
                                          load_embeddings=self.use_embeddings, working_dir=working_dir)

        self.opes = opes_factory(embed, rng, self.opes_type, self.model_name, self.output_type, working_dir)
        start_time = time.time()
        print('Loading pre-trained model...')
        self.opes.load_trained_best_model(metric_opt=self.metric_opt, custom_folder=self.custom_folder)
        time_model_loading = pd.DataFrame({'time': [time.time() - start_time]})
        print('Time to load model:', time_model_loading['time'][0], '[sec]')
        time_model_loading.to_csv(os.path.join(self.log_dir, self.policy_selection_name + '_model_loading_timing.csv'))

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
        auto_ope_es = AutoOPEEstimatorSelection(
            ope_estimators=self.ope_estimators,
            q_models=self.q_models,
            metrics=self.estimator_selection_metrics,
            data_type='real',
            random_state=self.random_state,
            i_task=i_task,
            partial_res_file_name_root=self.get_partial_res_file_path_root() + policy_name + '_'
        )
        auto_ope_es.set_black_box_params(self.opes, high_score_better=self.output_type.high_score_better)
        auto_ope_es.set_real_data(log_data=log_data_train, pi_e_dist=pi_e_dist_train[policy_name])
        auto_ope_es.evaluate_estimators(n_bootstrap=n_bootstrap, n_jobs=self.inner_n_jobs)
        return auto_ope_es

    def evaluate_policies(self, n_inference_bootstrap=1, n_task=10, test_ratio=0.5, outer_n_jobs=-1, inner_n_jobs=-1):
        """for set data, we evaluate policies (with bootstrapping estimator selection) for several times

        Args:
            n_inference_bootstrap (int): The number of bootstrap sampling in ope estimator selection.
                If None, we use original data only once.
            n_task (int, optional): Defaults to 10. The number of policy selection.
            test_ratio (float, optional): Defaults to 0.5. Ratio of test set. This is valid when real data are used.
            outer_n_jobs:
            inner_n_jobs:
        """
        self._prepare_model()
        super().evaluate_policies(n_inference_bootstrap, n_task, test_ratio, outer_n_jobs, inner_n_jobs)

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
