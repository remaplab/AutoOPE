import pickle

import pandas as pd

from black_box.data.real_ope_data import RealOffPolicyContextBanditData
from common.estimator_selection.base_estimator_selection import BaseEstimatorSelection


class AutoOPEEstimatorSelection(BaseEstimatorSelection):
    """
    Automatic off-policy estimator selection method for OPE
    """

    def __init__(self, ope_estimators, q_models, metrics='mse', data_type='synthetic', random_state=None, i_task=0,
                 partial_res_file_name_root='./black-box'):
        super().__init__(ope_estimators, q_models, metrics, data_type, random_state, i_task,
                         partial_res_file_name_root)
        self.model = None
        self.high_score_better = None
        self.backend = 'threads'

    def set_black_box_params(self, model, high_score_better):
        self.model = model
        self.high_score_better = high_score_better

    def evaluate_estimators_single_bootstrap_iter(self, log_data, pi_e_dist):
        """For given batch data, we estimate estimator performance

        Args:
            log_data (dict): batch bandit feedback
            pi_e_dist (np.array): action distribution by evaluation policy

        Returns:
            dict: key:estimator name, value:metric of estimator performance
        """

        mean_estimator_performance_dict = {}
        dataset = RealOffPolicyContextBanditData()
        x_to_pred = pd.DataFrame.from_dict(dataset.feature_engineering(log_data=log_data, cf_pi_b=pi_e_dist))
        estimator_performance = self._get_model_prediction(x_to_pred)
        mean_estimator_performance_dict.update(estimator_performance)

        mean_estimator_performance_df = pd.DataFrame({
            'estimator_name': mean_estimator_performance_dict.keys(),
            self.metrics: mean_estimator_performance_dict.values()
        })

        return self.get_single_bootstrap_performance(mean_estimator_performance_df)



    def _get_model_prediction(self, x_to_pred: pd.DataFrame) -> dict:
        estimator_performance = self.model.predict_score(x_to_pred)
        if self.high_score_better:
            estimator_performance = -estimator_performance
        # estimator_performance = self.model.predict_rank_pos(x_to_pred, True)
        estimator_performance = estimator_performance.to_dict(orient='list')
        estimator_performance_dict = {}
        for key, val_list in estimator_performance.items():
            estimator_performance_dict[key] = val_list[0]
        return estimator_performance_dict

    def _hyperparameters_tuning(self):
        return

    def save_mem_optimized(self, per_policy_es_path):
        log_data = self.log_data
        pi_e_dist = self.pi_e_dist
        model = self.model

        self.log_data = None
        self.pi_e_dist = None
        self.model = None

        pickle.dump(self, open(per_policy_es_path, 'wb'))

        self.model = model
        self.log_data = log_data
        self.pi_e_dist = pi_e_dist
