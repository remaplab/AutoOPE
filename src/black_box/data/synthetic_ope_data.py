from copy import deepcopy
from math import log10
from typing import List, Callable, Optional, Union

import pandas as pd
from obp.dataset.reward_type import RewardType
from obp.dataset.synthetic import polynomial_behavior_policy, linear_behavior_policy, \
    logistic_sparse_reward_function, logistic_polynomial_reward_function, logistic_reward_function, \
    SyntheticBanditDataset
from obp.ope import SelfNormalizedDoublyRobust as SNDR, \
    InverseProbabilityWeighting as IPW, \
    InverseProbabilityWeightingTuning as IPWTuning, \
    SelfNormalizedInverseProbabilityWeighting as SNIPW, \
    DoublyRobust as DR, \
    DoublyRobustTuning as DRTuning, \
    DirectMethod as DM, \
    SwitchDoublyRobust as SwitchDR, \
    SwitchDoublyRobustTuning as SwitchDRTuning, \
    DoublyRobustWithShrinkage as DRos, \
    DoublyRobustWithShrinkageTuning as DRosTuning, \
    SubGaussianDoublyRobust as SGDR, \
    SubGaussianDoublyRobustTuning as SGDRTuning, \
    SubGaussianInverseProbabilityWeighting as SGIPW, \
    SubGaussianInverseProbabilityWeightingTuning as SGIPWTuning, \
    OffPolicyEvaluation, RegressionModel, BaseOffPolicyEstimator
from obp.types import BanditFeedback
from sklearn.base import BaseEstimator

from black_box.common.constants import OP_ESTIMATOR_COL_NAME
from black_box.data.base_ope_data import BaseOffPolicyContextBanditData
from black_box.data.stats import *
from common.data.counterfactual_pi_b import get_counterfactual_bandit_feedback, get_counterfactual_pscore
from common.data.multiple_beta_synthetic_data import MultipleBetaSyntheticData


class SyntheticOffPolicyContextBanditData(BaseOffPolicyContextBanditData):
    def __init__(self, n_rounds: int, n_actions: int, dim_context: int = 1, reward_type: str = RewardType.BINARY.value,
                 reward_function: Union[str, Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 reward_std: float = 1.0, action_context: Optional[np.ndarray] = None,
                 behavior_policy_function: Union[str, Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 beta: Union[List[float], float] = 1.0, n_deficient_actions: int = 0,
                 dataset_name: str = 'synthetic_bandit_dataset',
                 cf_policy_function: Union[str, Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 cf_beta: float = 1.0, cf_n_deficient_actions: int = 0, random_state: int = None, n_bootstrap: int = 1,
                 gt_points=100000, gt_subsamples=10):
        self.n_rounds = int(n_rounds)
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.cf_n_deficient_actions = int(cf_n_deficient_actions)
        self.cf_policy_function = self._get_policy_function(cf_policy_function)
        self.cf_beta = cf_beta
        clazz = MultipleBetaSyntheticData if isinstance(beta, list) else SyntheticBanditDataset
        self.log_distribution = clazz(n_actions=int(n_actions),
                                      dim_context=int(dim_context),
                                      reward_type=str(reward_type),
                                      reward_function=self._get_reward_function(reward_function),
                                      reward_std=float(reward_std),
                                      action_context=action_context,
                                      behavior_policy_function=self._get_policy_function(behavior_policy_function),
                                      beta=beta,
                                      n_deficient_actions=int(n_deficient_actions),
                                      random_state=self.random_state,
                                      dataset_name=str(dataset_name))
        self.true_policy_value = self._calculate_ground_truth_policy_value(gt_points, gt_subsamples)

    @staticmethod
    def _get_policy_function(policy_function: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]):
        if callable(policy_function):
            return policy_function
        elif policy_function == 'polynomial_behavior_policy':
            return polynomial_behavior_policy
        elif policy_function == 'linear_behavior_policy':
            return linear_behavior_policy
        else:
            return None

    @staticmethod
    def _get_reward_function(reward_function: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]):
        if callable(reward_function):
            return reward_function
        elif reward_function == 'logistic_reward_function':
            return logistic_reward_function
        elif reward_function == 'logistic_polynomial_reward_function':
            return logistic_polynomial_reward_function
        elif reward_function == 'logistic_sparse_reward_function':
            return logistic_sparse_reward_function
        else:
            return None

    def generation(self, ope_estimators: list[BaseOffPolicyEstimator], binary_rwd_models, continuous_rwd_models,
                   compute_ope_perf: bool):
        boot_features, boot_se, boot_rel_ee = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        boot_errors, error_type, err = [], None, False
        rwd_models = binary_rwd_models if self.log_distribution.reward_type == RewardType.BINARY.value else \
            continuous_rwd_models

        for ii in range(self.n_bootstrap):
            cf_data = get_counterfactual_bandit_feedback(dataset=self.log_distribution,
                                                         cf_beta=self.cf_beta,
                                                         cf_policy_function=self.cf_policy_function,
                                                         cf_n_deficient_actions=self.cf_n_deficient_actions,
                                                         n_rounds=self.n_rounds)
            log_data = self.log_distribution.obtain_batch_bandit_feedback(self.n_rounds)
            cf_data['pscore'] = get_counterfactual_pscore(pi_e=cf_data['pi_b'], log_actions=log_data['action'],
                                                          log_positions=log_data['position'])
            cf_data['action'], cf_data['position'] = log_data['action'], log_data['position']
            cf_data['reward'] = None

            features = pd.DataFrame.from_dict(self._feature_engineering(log_data, cf_data))  # vector (1 x features)
            boot_features = pd.concat((boot_features, features))

            if compute_ope_perf:
                se, rel_ee = pd.DataFrame(index=[0]), pd.DataFrame(index=[0])
                try:
                    se, rel_ee, error_type = self._ope_performance(log_data=log_data, cf_pi_e=cf_data['pi_b'],
                                                                   ope_estimators=ope_estimators, rwd_models=rwd_models)
                    boot_se = pd.concat((boot_se, se))
                    boot_rel_ee = pd.concat((boot_rel_ee, rel_ee))
                    boot_errors.append(error_type)
                    if error_type is not None:
                        err = True
                except:
                    boot_se = pd.concat((boot_se, se))
                    boot_rel_ee = pd.concat((boot_rel_ee, rel_ee))
                    boot_errors.append('unknown')
                    err = True
                    continue

        if err:
            print('Error', boot_errors)

        return boot_features, boot_se, boot_rel_ee, boot_errors

    @staticmethod
    def average_on_bootstrap(boot_features, boot_rel_ee, boot_se):
        boot_features.reset_index(inplace=True, drop=True)
        boot_se.reset_index(inplace=True, drop=True)
        boot_rel_ee.reset_index(inplace=True, drop=True)

        categorical_df = boot_features.select_dtypes(exclude=np.number).drop_duplicates()
        assert categorical_df.shape[0] <= 1, "Error: Multiple values for categorical attributes in bootstrap"

        avg_boot_features = boot_features.mean(numeric_only=True).to_frame().T
        avg_boot_se = boot_se.mean(numeric_only=True).to_frame().T
        avg_boot_rel_ee = boot_rel_ee.mean(numeric_only=True).to_frame().T

        avg_boot_features = pd.concat([avg_boot_features, categorical_df], axis=1)

        return avg_boot_features, avg_boot_rel_ee, avg_boot_se

    def _ope_performance(self,
                         log_data: BanditFeedback,
                         cf_pi_e: np.ndarray,
                         ope_estimators: List[BaseOffPolicyEstimator],
                         rwd_models: Optional[List[BaseEstimator.__class__]] = None
                         ) -> (List[pd.DataFrame], List[pd.DataFrame]):
        all_se = pd.DataFrame(dtype=np.float64)
        all_rel_ee = pd.DataFrame(dtype=np.float64)
        err = None
        if self.true_policy_value == 0.0:
            err = 'gt=0.0'
        if (log_data['reward'] == 0).all():
            estimated_rewards_by_reg_model = np.zeros_like(log_data['pi_b'])
            all_zeros_or_ones_rwd = True
            err = 'gt=0.0' if self.true_policy_value == 0.0 else '0'
        elif (log_data['reward'] == 1).all():
            estimated_rewards_by_reg_model = np.ones_like(log_data['pi_b'])
            all_zeros_or_ones_rwd = True
            err = '1'
        else:
            all_zeros_or_ones_rwd = False
            estimated_rewards_by_reg_model = None

        for i, q_model in enumerate(rwd_models):
            renamed_ope_estimators = []

            for ope_estimator in deepcopy(ope_estimators):
                if 'IPW' in ope_estimator.estimator_name.upper():
                    if i == 0:
                        renamed_ope_estimators.append(ope_estimator)
                    else:
                        pass
                else:
                    ope_estimator.estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                    renamed_ope_estimators.append(ope_estimator)

            if not all_zeros_or_ones_rwd:
                q_model_instance = self.get_q_model_instance(log_data['n_rounds'], q_model)
                len_list = 1 if log_data['position'] is None else int(np.max(log_data['position']) + 1)

                regression_model = RegressionModel(
                    n_actions=self.log_distribution.n_actions,
                    len_list=len_list,
                    action_context=self.log_distribution.action_context,
                    base_model=q_model_instance,
                    fitting_method='normal'
                )
                try:
                    estimated_rewards_by_reg_model = regression_model.fit_predict(
                        context=log_data["context"],
                        action=log_data["action"],
                        reward=log_data["reward"],
                        pscore=log_data['pscore'],
                        position=log_data['position'],
                        action_dist=log_data['pi_b'],
                        n_folds=3,  # use 3-fold cross-fitting
                        random_state=self.random_state,
                    )
                except Exception as e:
                    print(e)
                    err = 'folds'
                    estimated_rewards_by_reg_model = regression_model.fit_predict(
                        context=log_data["context"],
                        action=log_data["action"],
                        reward=log_data["reward"],
                        pscore=log_data['pscore'],
                        position=log_data['position'],
                        action_dist=log_data['pi_b'],
                        n_folds=1,  # use 3-fold cross-fitting
                        random_state=self.random_state,
                    )
            ope = OffPolicyEvaluation(bandit_feedback=log_data, ope_estimators=renamed_ope_estimators)

            estimation_error_se_df = ope.summarize_estimators_comparison(
                ground_truth_policy_value=self.true_policy_value,
                action_dist=cf_pi_e,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                metric='se',
            ).T
            estimation_error_rel_ee_df = ope.summarize_estimators_comparison(
                ground_truth_policy_value=self.true_policy_value,
                action_dist=cf_pi_e,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                metric='relative-ee',
            ).T
            all_se = pd.concat((all_se, estimation_error_se_df), axis=1)
            all_rel_ee = pd.concat((all_rel_ee, estimation_error_rel_ee_df), axis=1)
        return all_se, all_rel_ee, err

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

    def _calculate_ground_truth_policy_value(self, gt_points, gt_subsamples) -> float:
        gt = []
        for subsample_idx in range(gt_subsamples):
            cf_batch_bandit_feedback_big_n = get_counterfactual_bandit_feedback(
                dataset=self.log_distribution, cf_beta=self.cf_beta, cf_policy_function=self.cf_policy_function,
                cf_n_deficient_actions=self.cf_n_deficient_actions, n_rounds=int(gt_points / gt_subsamples)
            )
            # To use always different data we discard the data used in the previous lines
            gt.append(self.log_distribution.calc_ground_truth_policy_value(
                expected_reward=cf_batch_bandit_feedback_big_n["expected_reward"],
                action_dist=cf_batch_bandit_feedback_big_n['pi_b']
            ))
            self.log_distribution.obtain_batch_bandit_feedback(int(gt_points / gt_subsamples))

        gt = np.mean(gt)
        return float(gt)

    @classmethod
    def get_estimator_features(cls, ope_estimators: list[BaseOffPolicyEstimator],
                               q_models: List[BaseEstimator.__class__]) -> pd.DataFrame:
        use_rwd_models, which_rewards_model = [], []
        names, is_self_norm, use_ipw, is_gaussian, is_switch, is_shrinked = [], [], [], [], [], []

        self_norm_est_classes = [SNIPW, SNDR]
        ipw_est_classes = [IPW, IPWTuning, SGIPW, SGIPWTuning, SNIPW, DR, DRTuning, DRos, DRosTuning, SwitchDR,
                           SwitchDRTuning, SGDR, SGDRTuning, SNDR]
        rwd_model_est_classes = [DR, DRTuning, DRos, DRosTuning, SwitchDR, SwitchDRTuning, SGDR, SGDRTuning, SNDR, DM]
        gaussian_est_classes = [SGDR, SGDRTuning, SGIPW, SGIPWTuning]
        switch_est_classes = [SwitchDR, SwitchDRTuning]
        shriked_est_classes = [DRos, DRosTuning]

        ope_estimators = cls._get_all_renamed_estimators(ope_estimators, q_models)

        for ope_e in ope_estimators:
            use_rwd_model = ope_e.__class__ in rwd_model_est_classes
            rwd_model_name = None
            if use_rwd_model:
                for q_model_name in [q_model.__name__ for q_model in q_models]:
                    if q_model_name in ope_e.estimator_name:
                        rwd_model_name = q_model_name

            names.append(ope_e.estimator_name)
            use_rwd_models.append(use_rwd_model)
            is_self_norm.append(ope_e.__class__ in self_norm_est_classes)
            use_ipw.append(ope_e.__class__ in ipw_est_classes)
            which_rewards_model.append(rwd_model_name)
            is_gaussian.append(ope_e.__class__ in gaussian_est_classes)
            is_switch.append(ope_e.__class__ in switch_est_classes)
            is_shrinked.append(ope_e.__class__ in shriked_est_classes)

        return pd.DataFrame.from_dict({OP_ESTIMATOR_COL_NAME: names,
                                       "is_self_norm": is_self_norm,
                                       "use_ipw": use_ipw,
                                       "use_rwd_models": use_rwd_models,
                                       "which_rwd_model": which_rewards_model,
                                       "is_gaussian": is_gaussian,
                                       "is_switch": is_switch,
                                       "is_shrinked": is_shrinked})

    @classmethod
    def _get_all_renamed_estimators(cls, ope_estimators, q_models):
        renamed_ope_estimators = []
        for i, q_model in enumerate(q_models):
            for ope_estimator in deepcopy(ope_estimators):
                if 'IPW' in ope_estimator.estimator_name.upper():
                    if i == 0:
                        renamed_ope_estimators.append(ope_estimator)
                    else:
                        pass
                else:
                    ope_estimator.estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                    renamed_ope_estimators.append(ope_estimator)
        return renamed_ope_estimators
