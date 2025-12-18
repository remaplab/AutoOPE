import pickle

from common.evaluation.base_evaluation import BaseEvaluation
from pasif.estimator_selection.conventional_estimator_selection import ConventionalEstimatorSelection



class RealDataEvaluation(BaseEvaluation):

    def __init__(self, ope_estimators, q_models, log_dir_path, log_bandit_feedback, eval_bandit_feedback, pi_e,
                 test_ratio=0.5, n_data_generation=10, random_state=None, estimator_selection_metrics='mse',
                 outer_n_jobs=-1, inner_n_jobs=None, n_bootstrap=10, undersampling_ratio=1.0, stratify: bool = True,
                 outer_n_jobs_gt=1):
        super().__init__(ope_estimators=ope_estimators, q_models=q_models, log_dir_path=log_dir_path,
                         test_ratio=test_ratio, n_data_generation=n_data_generation, random_state=random_state,
                         estimator_selection_metrics=estimator_selection_metrics, pi_e=pi_e, outer_n_jobs=outer_n_jobs,
                         inner_n_jobs=inner_n_jobs, outer_n_jobs_gt=outer_n_jobs_gt, n_bootstrap=n_bootstrap,
                         stratify=stratify)
        self.log_bandit_feedback = log_bandit_feedback
        self.eval_bandit_feedback = eval_bandit_feedback
        self.data_type = 'real'
        self.undersampling_ratio = undersampling_ratio



    def get_gt_policy(self, policy, policy_name, n_sampling=1):
        print('Ground truth computation, policy:', policy_name)
        c_es = ConventionalEstimatorSelection(ope_estimators=self.ope_estimators, q_models=self.q_models,
                                              stratify=self.stratify, metrics=self.estimator_selection_metrics,
                                              data_type=self.data_type, random_state=self.random_state)
        c_es.set_real_data(batch_bandit_feedback_1=self.eval_bandit_feedback[policy_name],
                           batch_bandit_feedback_2=self.log_bandit_feedback, action_dist_1_by_2=None,
                           action_dist_2_by_1=policy[0], evaluation_data='1')
        c_es.evaluate_estimators(n_inner_bootstrap=None, n_outer_bootstrap=None, n_jobs=1,
                                 outer_repeat_type=None, ground_truth_method='on_policy')
        policy_value = self.eval_bandit_feedback[policy_name]['reward'].mean()
        return c_es, policy_value



    def set_data(self, ps):
        ps.set_real_data(log_data=self.log_bandit_feedback, pi_e_distributions=self.pi_e,
                         undersampling_ratio=self.undersampling_ratio)



    def save_mem_optimized(self):
        pi_e = self.pi_e
        log_bandit_feedback = self.log_bandit_feedback
        eval_bandit_feedback = self.eval_bandit_feedback
        self.pi_e = None
        self.log_bandit_feedback = None
        self.eval_bandit_feedback = None
        pickle.dump(self, open(self.log_dir_path_save + '/evaluation_of_selection_method.pickle', 'wb'))
        self.pi_e = pi_e
        self.log_bandit_feedback = log_bandit_feedback
        self.eval_bandit_feedback = eval_bandit_feedback
