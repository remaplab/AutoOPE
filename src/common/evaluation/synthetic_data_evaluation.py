import pickle

from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from torch import optim

from common.evaluation.base_evaluation import BaseEvaluation
from common.evaluation.evaluation_utils import LearnedBehaviorPolicy
from black_box.policy_selection.auto_ope_policy_selection import AutoOPEPolicySelection
from common.data.multiple_beta_synthetic_data import MultipleBetaSyntheticData
from pasif.estimator_selection.conventional_estimator_selection import ConventionalEstimatorSelection
from pasif.policy_selection.conventional_policy_selection import ConventionalPolicySelection
from pasif.policy_selection.pasif_policy_selection import PASIFPolicySelection


class SyntheticDataEvaluation(BaseEvaluation):
    """
    Using synthetic data, we evaluate and compare estimator/policy selection methods.
    """

    def __init__(self, ope_estimators, q_models, log_dir_path, estimator_selection_metrics='mse', n_actions=5,
                 dim_context=5, beta_1=3, beta_2=7, reward_type='binary', reward_function=logistic_reward_function,
                 n_rounds_1=1000, n_rounds_2=1000, n_gt_sampling=100, test_ratio=0.5, pi_e={}, n_data_generation=10,
                 random_state=None, outer_n_jobs=-1, inner_n_jobs=None, n_bootstrap=10, outer_n_jobs_gt=1,
                 stratify: bool = True):
        """
        Set basic settings

        Args:
            ope_estimators (list): list of candidate estimators
            q_models (list): list of reward estimators used in model-depending estimators
            estimator_selection_metrics (str, optional): _description_. Defaults to 'mse'.
            n_actions (int, optional): Defaults to 5. Number of actions.
            dim_context (int, optional): Defaults to 5. Dim of context.
            beta_1 (int, optional): Defaults to 3. Beta for partial log data.
            beta_2 (int, optional): Defaults to 7. Beta for partial log data.
            reward_type (str, optional): Defaults to 'binary'. binary or continuous.
            reward_function (_type_, optional): Defaults to logistic_reward_function. mean reward function.
            n_rounds_1 (int, optional): Defaults to 1000. sample size of partial data 1.
            n_rounds_2 (int, optional): Defaults to 1000. sample size of partial data 2.
            test_ratio (float, optional): Defaults to 0.5. Ratio of test set.
                                          If None, we use whole data for both estimator selection and policy selection
                                          (valid when outer_loop_type='generation').
            pi_e (dict, optional): dictionary of evaluation policies. keys: name of policy (str). values: tuple
            including info of evaluation policy.
                                   ex. {'beta_1.0':('beta', 1.0), 'function_1':('function', pi(a|x), tau)}
                                   * ('beta', 1.0) (using beta to specify evaluation policy)
                                   * ('function', pi(a|x), tau) (Give any function as evaluation policy. To get
                                   action_diost, we use predict_proba(tau=tau))
            n_data_generation (int, optional): Defaults to 10. The number of policy selection.
            random_state (int, optional): Defaults to None.
        """
        super().__init__(ope_estimators=ope_estimators, q_models=q_models, log_dir_path=log_dir_path,
                         test_ratio=test_ratio, n_data_generation=n_data_generation, random_state=random_state,
                         estimator_selection_metrics=estimator_selection_metrics, pi_e=pi_e, outer_n_jobs=outer_n_jobs,
                         inner_n_jobs=inner_n_jobs, n_bootstrap=n_bootstrap, outer_n_jobs_gt=outer_n_jobs_gt,
                         stratify=stratify)
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.reward_type = reward_type
        self.reward_function = reward_function
        self.n_rounds_1 = n_rounds_1
        self.n_rounds_2 = n_rounds_2
        self.outer_loop_type = 'generation'
        self.data_type = 'synthetic'
        self.n_gt_sampling = n_gt_sampling



    def get_gt_policy(self, pi_e, policy_name):
        # {'beta_1.0':('beta', 1.0), 'function_1':('function', pi(a|x), tau)}
        # ground truth of estimator performance
        c_es = ConventionalEstimatorSelection(ope_estimators=self.ope_estimators, q_models=self.q_models,
                                              metrics=self.estimator_selection_metrics, stratify=self.stratify,
                                              data_type=self.data_type, random_state=self.random_state)
        if pi_e[0] == 'beta':
            cf_dataset = SyntheticBanditDataset(n_actions=self.n_actions, dim_context=self.dim_context,
                                                beta=pi_e[1],
                                                reward_type=self.reward_type,
                                                reward_function=self.reward_function,
                                                behavior_policy_function=None, random_state=self.random_state)
        elif pi_e[0] == 'function':
            learned_policy = LearnedBehaviorPolicy(pi_e[1])
            cf_dataset = SyntheticBanditDataset(
                n_actions=self.n_actions, dim_context=self.dim_context, beta=1.0 / pi_e[2],  # 1.0/tau == beta
                reward_type=self.reward_type, reward_function=self.reward_function,
                behavior_policy_function=learned_policy.behavior_policy_function_predict_proba,
                random_state=self.random_state)
        else:
            raise NotImplementedError(f"Policy selection method {pi_e[0]} not implemented.")

        log_dataset = MultipleBetaSyntheticData(
            n_actions=self.n_actions,
            dim_context=self.dim_context,
            beta=[self.beta_1, self.beta_2],
            reward_type=self.reward_type,
            reward_function=self.reward_function,
            behavior_policy_function=None,
            random_state=self.random_state
        )
        test_ratio = 1. if self.test_ratio is None else self.test_ratio
        c_es.set_synthetic_data(dataset_1=cf_dataset, n_rounds_1=1000000, dataset_2=log_dataset,
                                n_rounds_2=int(test_ratio * (self.n_rounds_1 + self.n_rounds_2)),
                                evaluation_data='1')
        c_es.evaluate_estimators(n_inner_bootstrap=None, n_outer_bootstrap=self.n_gt_sampling, n_jobs=self.outer_n_jobs,
                                 outer_repeat_type=self.outer_loop_type, ground_truth_method='ground-truth')

        batch_with_large_n = cf_dataset.obtain_batch_bandit_feedback(n_rounds=1000000)
        policy_value = cf_dataset.calc_ground_truth_policy_value(
            expected_reward=batch_with_large_n['expected_reward'], action_dist=batch_with_large_n['pi_b'])

        return c_es, policy_value



    def evaluate_conventional_selection_method(self, load):
        """
        evaluate conventional estimator/policy selection
        """
        conventional_ps = None
        if not load:
            dataset_1 = SyntheticBanditDataset(n_actions=self.n_actions, dim_context=self.dim_context, beta=self.beta_1,
                                               reward_type=self.reward_type, reward_function=self.reward_function,
                                               behavior_policy_function=None, random_state=self.random_state)
            dataset_2 = SyntheticBanditDataset(n_actions=self.n_actions, dim_context=self.dim_context, beta=self.beta_2,
                                               reward_type=self.reward_type, reward_function=self.reward_function,
                                               behavior_policy_function=None, random_state=self.random_state)

            conventional_ps = ConventionalPolicySelection(ope_estimators=self.ope_estimators, q_models=self.q_models,
                                                          estimator_selection_metrics=self.estimator_selection_metrics,
                                                          data_type=self.data_type, random_state=self.random_state,
                                                          stratify=self.stratify)
            conventional_ps.set_synthetic_data(dataset_1=dataset_1, n_rounds_1=self.n_rounds_1,
                                               dataset_2=dataset_2, n_rounds_2=self.n_rounds_2,
                                               pi_e=self.pi_e, evaluation_data='partial_random')

        self.policy_estimator_selection_evaluation(conventional_ps, load, 'conventional')



    def set_data(self, ps):
        dataset = MultipleBetaSyntheticData(n_actions=self.n_actions, dim_context=self.dim_context,
                                            beta=[self.beta_1, self.beta_2], reward_type=self.reward_type,
                                            reward_function=self.reward_function, behavior_policy_function=None,
                                            random_state=self.random_state)
        ps.set_synthetic_data(synth_dataset=dataset, n_rounds=self.n_rounds_1 + self.n_rounds_2, pi_e_params=self.pi_e)



    def set_pasif_method_params_wrapper(self, arguments):
        policy_name_for_key = None
        pasif_optimizer_dict = {'0': optim.SGD, '1': optim.Adam}
        method_param_dict = {}
        for model_pi_e, beta, pasif_k, pasif_rw, pasif_bs, pasif_n_eopchs, pasif_opt, pasif_lr in \
                zip(arguments.model_list_for_pi_e, arguments.beta_list_for_pi_e, arguments.pasif_k,
                    arguments.pasif_regularization_weight, arguments.pasif_batch_size, arguments.pasif_n_epochs,
                    arguments.pasif_optimizer, arguments.pasif_lr):
            if model_pi_e == 0:
                policy_name_for_key = 'beta_' + str(beta)
            elif model_pi_e == 1:
                policy_name_for_key = 'model_ipw_lr_beta_' + str(beta)
            elif model_pi_e == 2:
                policy_name_for_key = 'model_ipw_rf_beta_' + str(beta)
            elif model_pi_e == 3:
                if arguments.reward_type == 'binary':
                    policy_name_for_key = 'model_qlr_lr_beta_' + str(beta)
                elif arguments.reward_type == 'continuous':
                    policy_name_for_key = 'model_qlr_rr_beta_' + str(beta)
            elif model_pi_e == 4:
                policy_name_for_key = 'model_qlr_rf_beta_' + str(beta)

            method_param_dict[policy_name_for_key] = {
                'k': pasif_k,
                'regularization_weight': pasif_rw,
                'batch_size': pasif_bs,
                'n_epochs': pasif_n_eopchs,
                'optimizer': pasif_optimizer_dict[str(pasif_opt)],
                'lr': pasif_lr
            }
        self.set_pasif_method_params(method_name='pasif', method_params=method_param_dict)



    def save_mem_optimized(self):
        pickle.dump(self, open(self.log_dir_path_save + '/evaluation_of_selection_method.pickle', 'wb'))
