import random
from abc import ABCMeta
from math import log10


class AbstractEstimatorSelection(metaclass=ABCMeta):
    def __init__(self, ope_estimators, q_models, random_state=None):
        self.q_models = q_models
        self.all_result = None
        self.summarized_result = None
        self.random_state = random.randint(1, 10000000) if random_state is None else random_state
        self.ope_estimators = ope_estimators

    def get_q_model_instance(self, batch_bandit_feedback_rounds, q_model):
        verbose = 0
        from lightgbm import LGBMClassifier
        if q_model is LGBMClassifier:
            verbose = -1

        q_model_instance = q_model(random_state=self.random_state)
        q_model_args = {'random_state': self.random_state, 'verbose': verbose, 'n_jobs': 1}
        #if hasattr(q_model_instance, 'solver') and batch_bandit_feedback_rounds > 5000:
         #   q_model_args['solver'] = 'saga'
          #  q_model_args['max_iter'] = pow(10, round(log10(batch_bandit_feedback_rounds)) - 1)
        q_model_instance = q_model(**q_model_args)
        return q_model_instance

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

    def get_best_estimator(self, rnd):
        """
        Get best ope estimator besed on mean metric

        Returns:
            obp.ope, sklearn-predictor: best ope estimator and q model
        """
        # define best estimator
        best_estimators = self.summarized_result['estimator_name'][self.summarized_result['rank'] == 1].values
        best_estimator_name = best_estimators[0]
        if len(best_estimators) > 1:
            best_estimator_name = rnd.choice(best_estimators)
        idx = best_estimator_name.find('_qmodel_')
        if idx == -1:
            best_ope_name = best_estimator_name
        else:
            best_ope_name = best_estimator_name[:idx]

        best_estimator = None
        for estimator in self.ope_estimators:
            if estimator.estimator_name == best_ope_name:
                best_estimator = estimator
        best_q_model = self.q_models[0]
        for q_model in self.q_models:
            if str(q_model) in best_estimator_name:
                best_q_model = q_model

        return best_estimator, best_q_model
