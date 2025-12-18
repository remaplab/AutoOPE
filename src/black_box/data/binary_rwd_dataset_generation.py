import argparse
from typing import Union

from obp.dataset import logistic_reward_function, logistic_polynomial_reward_function, logistic_sparse_reward_function, \
    linear_behavior_policy, polynomial_behavior_policy
from scipy.stats import randint, uniform

from common.ope_estimators import get_op_estimators
from black_box.data.ope_data_generator import DataGenerator


def main():
    # get command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', type=int, default=250000)
    parser.add_argument('--n_bootstrap', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--force_regeneration', action="store_true", default=False)
    parser.add_argument('--gt_points', type=int, default=100000)
    parser.add_argument('--gt_subsamples', type=int, default=10)
    arguments = parser.parse_args()

    hypara_dict, ope_estimators, q_models = get_op_estimators(reward_type='binary')

    bin_rwd_funcs = [logistic_reward_function, logistic_polynomial_reward_function, logistic_sparse_reward_function,
                     None]
    policy_funcs = [polynomial_behavior_policy, linear_behavior_policy, None]

    # Data Generation
    data_generator = DataGenerator(random_state=arguments.random_state, op_estimators=ope_estimators,
                                   n_jobs=arguments.n_jobs, binary_rwd_models=q_models)
    data_generator.set_nactions_distribution(distribution=randint, low=2, high=20)
    data_generator.set_nrounds_distribution(distribution=randint, low=100, high=8000)
    # data_generator.set_n_def_actions_distribution(distribution=betabinom, no_def_action_prob=0.8,
    #                                              upper_bound_param_name='n', a=1, b=4)
    # data_generator.set_op_n_def_actions_distribution(distribution=betabinom, no_def_action_prob=0.8,
    #                                                 upper_bound_param_name='n', a=1, b=4)
    data_generator.set_contexts_dims_distribution(distribution=randint, low=1, high=10)
    data_generator.set_binary_reward_probability(bin_perc=1.0)
    data_generator.set_rewards_distribution(binary_rwd_funcs=bin_rwd_funcs)
    data_generator.set_target_policies_distribution(policies_funcs=policy_funcs, betas_distr=uniform, loc=-10, scale=20)
    data_generator.set_behaviour_policies_distribution(policies_funcs=policy_funcs, betas_distr=uniform, loc=-10,
                                                       scale=20, multi_beta_prob=0.5)

    data_generator.generate_tuples(n_points=arguments.n_points, n_bootstrap=arguments.n_bootstrap,
                                   gt_points=arguments.gt_points, gt_subsamples=arguments.gt_subsamples,
                                   force_regeneration=arguments.force_regeneration, batch_size=arguments.batch_size)


if __name__ == "__main__":
    main()
