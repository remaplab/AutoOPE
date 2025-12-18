from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
from obp.ope import BaseOffPolicyEstimator
from obp.utils import check_ope_inputs, check_array, estimate_confidence_interval_by_bootstrap
import cvxpy as cp
from tqdm import tqdm

from common.regression_model_stratified import RegressionModelStratified



@dataclass
class OPERA(BaseOffPolicyEstimator):
    """
    This meta - estimator aggregates a set of base OPE estimators by computing optimal weights based on a bootstrapping
    procedure.It conforms to the Open Bandit Pipeline API.

    Parameters:
    base_estimators(list): List of base estimators(instances of BaseOffPolicyEstimator).
    B(int): Number of bootstrap iterations to use for weight estimation.
    eta(float): Bootstrap subsample fraction(the subsample size will be floor(eta * n)).
    random_state(Optional[int]): Random seed for reproducibility.
    """
    base_estimators: list
    q_models: list
    B: int = 100
    eta: float = 0.8
    random_state: Optional[int] = None
    estimator_name: str = "opera"
    n_jobs_fit = -1
    stratify: bool = True



    def __post_init__(self) -> None:
        if self.random_state is None:
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(self.random_state)



    def estimate_policy_value(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            action_dist: np.ndarray,
            estimated_rewards_by_reg_model: np.ndarray,
            pscore: Optional[np.ndarray] = None,
            position: Optional[np.ndarray] = None,
            estimated_pscore: Optional[np.ndarray] = None,
            context: Optional[np.ndarray] = None,
            pi_b: Optional[np.ndarray] = None,
            **kwargs,
    ) -> float:
        """
        Estimate the evaluation policy 's performance via an ensemble of base estimators.
        This method takes the same arguments as other OPE estimators:
        - reward: (n_rounds,) observed rewards.
        - action: (n_rounds,) actions taken.
        - action_dist: (n_rounds, n_actions, len_list) evaluation policy probabilities.
        - position: (n_rounds,) optional position indices.
        - ** kwargs: additional parameters(e.g., pscore, estimated_rewards_by_reg_model).

        Returns:
        A float representing the ensemble(OPERA) policy value estimate.
        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        check_ope_inputs(action_dist=action_dist, position=position, action=action, reward=reward)

        # policy value estimation of each estimator on entire logging data
        log_data = {
            'action_context': kwargs['action_embed'],
            'n_actions': action_dist.shape[1],
            'action': action,
            'reward': reward,
            'position': position,
            'pscore': pscore,
            'context': context,
            'pi_b': pi_b
        }
        full_estimates, full_estimates_dict = self.get_base_estimators_estimates(log_data, pi_e_dist=action_dist)

        # policy value estimation of each estimator on bootstrap sampled logging data
        n = reward.shape[0]
        n1 = max(1, int(self.eta * n))
        bootstrap_estimates = np.zeros((full_estimates.shape[0], self.B))
        for b in tqdm(range(self.B), desc="Bootstrapping"):
            indices = self._rng.choice(n, size=n1, replace=True)
            boot_log_data = {
                'n_actions': action_dist.shape[1],
                'action_context': kwargs['action_embed'],
                'action': action[indices],
                'reward': reward[indices],
                'position': position[indices],
                'pscore': pscore[indices],
                'context': context[indices],
                'pi_b': pi_b[indices]
            }
            boot_action_dist = action_dist[indices]
            b_estimate, b_estimate_dict = self.get_base_estimators_estimates(boot_log_data, pi_e_dist=boot_action_dist)
            bootstrap_estimates[:, b] = b_estimate

        # Compute delta: differences between bootstrap estimates and full-data estimates
        delta = bootstrap_estimates - full_estimates[:, np.newaxis]
        A_hat = n1 * (delta @ delta.T) / (self.B * n)

        # Solve for optimal weights (analytical solution): alpha = (A_hat^{-1} 1) / (1^T A_hat^{-1} 1)
        #try:
        #    inv_A = np.linalg.inv(A_hat)
        #except np.linalg.LinAlgError:
        #    print("Not Invertible A matrix|")
        #    inv_A = np.linalg.pinv(A_hat)
        #
        #ones_vec = np.ones(k)
        #alpha = inv_A.dot(ones_vec)
        #alpha /= np.sum(alpha)
        #self._last_weights = alpha  # Store weights for inspection, if needed.

        # Try cvxpy and see if same results
        alpha_cp = cp.Variable(A_hat.shape[0])
        constraints = [cp.sum(alpha_cp) == 1]
        objective = cp.Minimize(cp.quad_form(alpha_cp, A_hat))

        # Form and solve problem.
        prob = cp.Problem(objective, constraints)
        prob.solve()  # Returns the optimal value.
        if prob.status != "optimal":
            print("Status:", prob.status)

        ensemble_value = np.dot(alpha_cp.value, full_estimates)
        print(list(alpha_cp.value))
        return ensemble_value



    def get_base_estimators_estimates(self, log_data, pi_e_dist):
        full_estimates_dict = {}
        full_estimates = []

        for i, q_model in enumerate(self.q_models):
            renamed_ope_estimators = []

            for ope_estimator in deepcopy(self.base_estimators):
                if 'IPW' in ope_estimator.estimator_name:
                    if i == 0:
                        renamed_ope_estimators.append(ope_estimator)
                else:
                    new_estimator_name = ope_estimator.estimator_name + '_qmodel_' + q_model.__name__
                    ope_estimator.estimator_name = new_estimator_name
                    renamed_ope_estimators.append(ope_estimator)

            q_model_instance = self.get_q_model_instance(log_data, q_model)
            q_model_instance.n_jobs = self.n_jobs_fit
            len_list = 1 if log_data['position'] is None else int(log_data['position'].max() + 1)

            regression_model = RegressionModelStratified(
                n_actions=log_data['n_actions'],
                action_context=log_data['action_context'],
                base_model=q_model_instance,
                len_list=len_list,
                fitting_method='normal',
                stratify=self.stratify
            )
            estimated_rewards_by_reg_model_per_est = regression_model.fit_predict(
                context=log_data['context'],
                action=log_data['action'],
                reward=log_data['reward'],
                pscore=log_data['pscore'],
                position=log_data['position'],
                action_dist=log_data['pi_b'],
                n_folds=3,  # use 3-fold cross-fitting
                random_state=self.random_state
            )

            for j, estimator in enumerate(renamed_ope_estimators):
                estimate = estimator.estimate_policy_value(
                    reward=log_data['reward'],
                    action=log_data['action'],
                    position=log_data['position'],
                    pscore=log_data['pscore'],
                    estimated_pscore=None,
                    action_dist=pi_e_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model_per_est,
                )
                full_estimates_dict[estimator.estimator_name] = estimate
                full_estimates.append(estimate)

        return np.array(full_estimates), full_estimates_dict



    def estimate_interval(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            action_dist: np.ndarray,
            position: Optional[np.ndarray] = None,
            alpha: float = 0.05,
            n_bootstrap_samples: int = 10000,
            random_state: Optional[int] = None,
            **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Estimate a confidence interval for the ensemble policy value using bootstrap.
        This method uses the fixed weights computed on the full data to construct bootstrap ensemble estimates and then
        computes the confidence interval via percentiles.

        Returns: A dictionary containing the mean and the upper and lower confidence bounds.
        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        check_ope_inputs(action_dist=action_dist, position=position, action=action, reward=reward, **kwargs)

        # Ensure full-data ensemble weights are computed
        ensemble_full = self.estimate_policy_value(reward, action, action_dist, position=position, **kwargs)
        k = len(self.base_estimators)
        full_estimates = np.zeros(k)
        for i, estimator in enumerate(self.base_estimators):
            full_estimates[i] = estimator.estimate_policy_value(
                reward, action, action_dist, position=position, **kwargs
            )
        n = reward.shape[0]
        n1 = max(1, int(self.eta * n))
        bootstrap_estimates = np.zeros((k, n_bootstrap_samples))
        for b in range(n_bootstrap_samples):
            indices = self._rng.choice(n, size=n1, replace=True)
            boot_reward = reward[indices]
            boot_action = action[indices]
            boot_action_dist = action_dist[indices]
            boot_position = position[indices]
            boot_kwargs = {
                key: (value[indices] if isinstance(value, np.ndarray) and value.shape[0] == n else value)
                for key, value in kwargs.items()
            }
            for i, estimator in enumerate(self.base_estimators):
                bootstrap_estimates[i, b] = estimator.estimate_policy_value(
                    boot_reward, boot_action, boot_action_dist, position=boot_position, **boot_kwargs
                )
        # Use the fixed weights computed earlier (or recompute if not available)
        if hasattr(self, "_last_weights"):
            weights = self._last_weights
        else:
            # Recompute weights if necessary (using a smaller bootstrap if needed)
            delta_full = bootstrap_estimates - full_estimates[:, np.newaxis]
            A_hat = (n1 / n) * (delta_full @ delta_full.T) / n_bootstrap_samples
            try:
                inv_A = np.linalg.inv(A_hat)
            except np.linalg.LinAlgError:
                inv_A = np.linalg.pinv(A_hat)
            ones_vec = np.ones(k)
            weights = inv_A.dot(ones_vec)
            weights /= np.sum(weights)
        ensemble_bootstrap = np.dot(weights, bootstrap_estimates)
        return estimate_confidence_interval_by_bootstrap(
            samples=ensemble_bootstrap,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )



    def _estimate_round_rewards(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        OPERA does not directly support per - round reward estimation.Use the ensemble estimate via estimate_policy_value
        instead.
        """
        raise NotImplementedError("OPERA does not support individual round-wise reward estimation")



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