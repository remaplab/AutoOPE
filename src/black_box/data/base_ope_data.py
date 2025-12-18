from typing import Dict, List

import numpy as np
from obp.dataset.reward_type import RewardType
from obp.types import BanditFeedback
from scipy.stats import skew, kurtosis

from black_box.data.stats import kl_div, pearson_chi_squared_dist, inner_product_dist, chebyshev_dist, div, k_div, \
    kumar_johnson_dist, euclidian_dist, kulczynski_dist, city_block, total_variance_dist, neyman_chi_squared_dist, \
    additive_symmetric_chi_squared_dist, canberra_dist, jensen_shannon_dist

MAX_PS_SCORE = 10000


class BaseOffPolicyContextBanditData:
    def _feature_engineering(self, log_data: BanditFeedback, cf_data: BanditFeedback) -> Dict[str, List]:
        clipping = 10
        feature_point = {'n_samples': [self._get_n_samples(log_data)],
                         'n_actions': [self._get_n_actions(log_data)],
                         'n_def_actions': [self._get_n_def_actions(log_data)],
                         'context_dim': [self._get_context_dim(log_data)],
                         'avg_context_var': [self._get_avg_context_var(log_data)],
                         'action_var': [self._get_actions_var(log_data)],
                         'reward_type': [self._get_reward_type(log_data)],
                         'reward_std': [self._get_reward_sample_std(log_data)],
                         'reward_mean': [self._get_reward_sample_mean(log_data)],
                         'reward_skew': [self._get_reward_sample_skew(log_data)],
                         'reward_kurtosis': [self._get_reward_sample_kurtosis(log_data)],
                         'max_action_prob_log': [self._get_max_action_prob_log_policy(log_data)],
                         'min_action_prob_log': [self._get_min_action_prob_log_policy(log_data)],
                         'max_action_prob_cf': [self._get_max_action_prob_cf_policy(cf_data)],
                         'min_action_prob_cf': [self._get_min_action_prob_cf_policy(cf_data)],
                         'max_ps': [self._get_max_ps(log_data, cf_data)],
                         'self_norm_denor': [self._get_self_normalization_ps_factor(log_data, cf_data)],
                         'n_clipped_weights': [self._get_n_clipped_weights(log_data, cf_data, clipping)],
                         'total_var_dist': [self._get_total_variation_distance(log_data, cf_data)],
                         'pearson_chi_squared_dist': [self._get_pearson_chi_squared_dist(log_data, cf_data)],
                         'inner_product_dist': [self._get_inner_product_dist(log_data, cf_data)],
                         'chebyshev_dist': [self._get_chebyshev_dist(log_data, cf_data)],
                         'neyman_chi_squared_dist': [self._get_neyman_chi_squared_dist(log_data, cf_data)],
                         'div': [self._get_div(log_data, cf_data)],
                         'canberra': [self._get_canberra(log_data, cf_data)],
                         'k_div_log_cf': [self._get_k_div(log_data, cf_data)],
                         'k_div_cf_log': [self._get_k_div(cf_data, log_data)],
                         'jensen_shannon_dist': [self._get_jensen_shannon_dist(log_data, cf_data)],
                         'kl_divergence_cf_log': [self._get_kl_divergence_cf_log(log_data, cf_data)],
                         'kl_divergence_log_cf': [self._get_kl_divergence_log_cf(log_data, cf_data)],
                         'kumar_johnson_dist': [self._get_kumar_johnson_dist(log_data, cf_data)],
                         'additive_symmetric_chi_squared_dist': [self.get_additive_symmetric_chi_squared_dist(log_data, cf_data)],
                         'euclidian_dist': [self._get_euclidian_dist(log_data, cf_data)],
                         'kulczynski_dist': [self._get_kulczynski_dist(log_data, cf_data)],
                         'city_block': [self._get_city_block(log_data, cf_data)]}
        return feature_point

    def _get_n_samples(self, log_data: BanditFeedback) -> int:
        return len(log_data['action'])

    def _get_n_actions(self, log_data: BanditFeedback) -> int:
        return log_data['n_actions']

    def _get_n_def_actions(self, log_data: BanditFeedback) -> int:
        return log_data['n_actions'] - np.unique(log_data['action']).shape[0]

    def _get_context_dim(self, log_data: BanditFeedback) -> int:
        return log_data['context'].shape[1]  # maybe better: np.rank(self.log_data['context'])

    def _get_avg_context_var(self, log_data: BanditFeedback) -> float:
        avg_ctx_var = float(np.mean(np.var(log_data['context'], axis=0), axis=0))
        return avg_ctx_var

    def _get_actions_var(self, log_data: BanditFeedback) -> float:
        actions_var = float(np.var(log_data['action']))
        return actions_var

    def _get_reward_type(self,  log_data: BanditFeedback) -> str:
        return RewardType.BINARY.value if ((log_data['reward'] == 0) | (log_data['reward'] == 1)).all() \
            else RewardType.CONTINUOUS.value

    def _get_reward_sample_std(self, log_data: BanditFeedback) -> float:
        reward_sample_std = float(np.std(log_data['reward']))
        return reward_sample_std

    def _get_reward_sample_mean(self, log_data: BanditFeedback) -> float:
        reward_sample_mean = float(np.mean(log_data['reward']))
        return reward_sample_mean

    def _get_reward_sample_skew(self, log_data: BanditFeedback) -> float:
        skew_ = float(skew(log_data['reward']))
        return skew_

    def _get_reward_sample_kurtosis(self, log_data: BanditFeedback) -> float:
        kurtosis_ = float(kurtosis(log_data['reward']))
        return kurtosis_

    def _get_max_action_prob_log_policy(self, log_data: BanditFeedback) -> float:
        max_action_prob_log_policy = np.max(log_data['pi_b'].mean(axis=0))
        return max_action_prob_log_policy

    def _get_min_action_prob_log_policy(self, log_data: BanditFeedback) -> float:
        min_action_prob_log_policy = np.min(log_data['pi_b'].mean(axis=0))
        return min_action_prob_log_policy

    def _get_max_action_prob_cf_policy(self, cf_data: BanditFeedback) -> float:
        max_action_prob_target_policy = np.max(cf_data['pi_b'].mean(axis=0))
        return max_action_prob_target_policy

    def _get_min_action_prob_cf_policy(self, cf_data: BanditFeedback) -> float:
        min_action_prob_target_policy = np.min(cf_data['pi_b'].mean(axis=0))
        return min_action_prob_target_policy

    def _get_kl_divergence_cf_log(self, log_data: BanditFeedback, cf_data: BanditFeedback) -> float:
        kl_divergence_target_log = kl_div(cf_data['pi_b'], log_data['pi_b'])
        return kl_divergence_target_log

    def _get_kl_divergence_log_cf(self, log_data: BanditFeedback, cf_data: BanditFeedback) -> float:
        kl_divergence_log_target = kl_div(log_data['pi_b'], cf_data['pi_b'])
        return kl_divergence_log_target

    def _get_max_ps(self, log_data: BanditFeedback, cf_data: BanditFeedback) -> float:
        max_ps = np.max(np.divide(log_data['pscore'], cf_data['pscore']))
        return max_ps

    def _get_self_normalization_ps_factor(self, log_data: BanditFeedback, cf_data: BanditFeedback) -> float:
        self_normalization_ps_factor = float(np.mean(np.divide(cf_data['pscore'], log_data['pscore'])))
        return self_normalization_ps_factor

    def _get_n_clipped_weights(self, log_data: BanditFeedback, cf_data: BanditFeedback, clipping: int) -> int:
        num_of_clipped_weights = np.count_nonzero(clipping < np.divide(cf_data['pscore'], log_data['pscore']))
        return num_of_clipped_weights

    def _get_pearson_chi_squared_dist(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = pearson_chi_squared_dist(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_inner_product_dist(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = inner_product_dist(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_chebyshev_dist(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = chebyshev_dist(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_div(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = div(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_k_div(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = k_div(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_kumar_johnson_dist(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = kumar_johnson_dist(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_euclidian_dist(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = euclidian_dist(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_kulczynski_dist(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = kulczynski_dist(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_city_block(self, log_data: BanditFeedback, cf_data: BanditFeedback):
        dist = city_block(log_data['pi_b'], cf_data['pi_b'])
        return dist

    def _get_total_variation_distance(self, log_data: BanditFeedback, cf_data: BanditFeedback) -> float:
        total_variation_distance = total_variance_dist(log_data['pi_b'], cf_data['pi_b'])
        return total_variation_distance

    def get_additive_symmetric_chi_squared_dist(self, log_data, cf_data):
        additive_symmetric_chi_squared = additive_symmetric_chi_squared_dist(log_data['pi_b'], cf_data['pi_b'])
        return additive_symmetric_chi_squared

    def _get_neyman_chi_squared_dist(self, log_data, cf_data):
        neyman_chi_squared = neyman_chi_squared_dist(log_data['pi_b'], cf_data['pi_b'])
        return neyman_chi_squared

    def _get_canberra(self, log_data, cf_data):
        canberra = canberra_dist(log_data['pi_b'], cf_data['pi_b'])
        return canberra

    def _get_jensen_shannon_dist(self, log_data, cf_data):
        jensen_shannon = jensen_shannon_dist(log_data['pi_b'], cf_data['pi_b'])
        return jensen_shannon
