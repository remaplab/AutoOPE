from abc import ABCMeta, abstractmethod
from math import log10
import random


class AbstractPolicySelection(metaclass=ABCMeta):
    """
    Abstract Interface class for policy selection methods
    """

    def __init__(self, ope_estimators, q_models, stratify, estimator_selection_metrics='mse', data_type='synthetic',
                 random_state=None):
        assert estimator_selection_metrics == 'mse' or estimator_selection_metrics == 'mean relative-ee', \
            'estimator_selection_metrics must be mse or mean relative-ee'
        assert data_type == 'synthetic' or data_type == 'real', 'data_type must be synthetic or real'

        self.ope_estimators = ope_estimators
        self.q_models = q_models
        self.estimator_selection_metrics = estimator_selection_metrics
        self.data_type = data_type
        self.random_state = random.randint(1, 10000000) if random_state is None else random_state
        self.all_estimator_selection_result = None
        self.all_result = None
        self.summarized_result = None
        self.stratify = stratify

    @abstractmethod
    def evaluate_policies(self, n_inference_bootstrap, n_task, test_ratio, outer_n_jobs, inner_n_jobs):
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
        pass

    def get_all_estimator_selection_results(self):
        """
        Get all estimator selection results (all outer lopp results for all evaluation policy)

        Returns:
            dict: results for all outer loop. key: policy name, value: df.
        """
        return self.all_estimator_selection_result

    def get_all_results(self):
        """
        Get all evaluation results (all outer lopp results)

        Returns:
            pd.DataFrame: results for all outer loop
        """
        return self.all_result

    def get_summarized_results(self):
        """
        Get summarized evaluation results

        Returns:
            pd.DataFrame: results (mean results for outer loops)
        """
        return self.summarized_result

    def get_best_policy(self, rnd):
        """
        Get best policy

        Returns:
            str: name of best policy
        """
        best_policies = self.summarized_result['policy_name'][self.summarized_result['rank'] == 1].values
        best_policy_name = best_policies[0]
        if len(best_policies) > 1:
            best_policy_name = rnd.choice(best_policies)
        return best_policy_name

    def get_q_model_instance(self, batch_bandit_feedback_rounds, q_model):
        verbose = 0
        from lightgbm import LGBMClassifier
        if q_model is LGBMClassifier:
            verbose = -1

        #q_model_instance = q_model(random_state=self.random_state)
        q_model_args = {'random_state': self.random_state, 'verbose': verbose, 'n_jobs': 1}
        #if hasattr(q_model_instance, 'solver') and batch_bandit_feedback_rounds > 5000:
         #   q_model_args['solver'] = 'saga'
          #  q_model_args['max_iter'] = pow(10, round(log10(batch_bandit_feedback_rounds)) - 1)
        q_model_instance = q_model(**q_model_args)
        return q_model_instance
