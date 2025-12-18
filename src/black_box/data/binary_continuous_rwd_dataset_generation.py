import argparse

from obp.dataset import sparse_reward_function, polynomial_reward_function, linear_reward_function, \
    logistic_sparse_reward_function, logistic_polynomial_reward_function, logistic_reward_function, \
    linear_behavior_policy, polynomial_behavior_policy, linear_behavior_policy_logit
from obp.ope import InverseProbabilityWeighting, SelfNormalizedInverseProbabilityWeighting, DoublyRobust
from scipy.stats import randint, loguniform, uniform, betabinom
from sklearn.linear_model import LinearRegression, LogisticRegression

from black_box.data.ope_data_generator import DataGenerator


def main():
    # get command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', type=int, default=20000)
    parser.add_argument('--random_state', type=int, default=42)
    arguments = parser.parse_args()

    # Parameters for generation
    op_estimators = [InverseProbabilityWeighting(), SelfNormalizedInverseProbabilityWeighting(), DoublyRobust()]
    continuous_rewards_models = [LinearRegression]
    binary_rewards_models = [LogisticRegression]
    bin_rwd_funcs = [logistic_reward_function, logistic_polynomial_reward_function, logistic_sparse_reward_function,
                     None]
    cont_rwd_funcs = [linear_reward_function, polynomial_reward_function, sparse_reward_function, None]
    policy_funcs = [polynomial_behavior_policy, linear_behavior_policy, linear_behavior_policy_logit, None]

    # Data Generation
    data_generator = DataGenerator(random_state=arguments.random_state, op_estimators=op_estimators,
                                   binary_rwd_models=binary_rewards_models,
                                   continuous_rwd_models=continuous_rewards_models)
    data_generator.set_nactions_distribution(distribution=randint, low=2, high=10)
    data_generator.set_nrounds_distribution(distribution=randint, low=100, high=1000)
    data_generator.set_n_def_actions_distribution(distribution=betabinom, no_def_action_prob=0.8,
                                                  upper_bound_param_name='n', a=1, b=4)
    data_generator.set_op_n_def_actions_distribution(distribution=betabinom, no_def_action_prob=0.8,
                                                     upper_bound_param_name='n', a=1, b=4)
    data_generator.set_contexts_dims_distribution(distribution=randint, low=2, high=10)
    data_generator.set_binary_reward_probability(bin_perc=0.5)
    data_generator.set_rewards_distribution(binary_rwd_funcs=bin_rwd_funcs, continuous_rwd_funcs=cont_rwd_funcs,
                                            rwd_std_distr=loguniform, a=1, b=10)
    data_generator.set_behaviour_policies_distribution(policies_funcs=policy_funcs, betas_distr=uniform, loc=-10,
                                                       scale=20)
    data_generator.set_target_policies_distribution(policies_funcs=policy_funcs, betas_distr=uniform, loc=-10, scale=20)

    data_generator.generate_tuples(n_points=arguments.n_points, n_bootstrap=arguments.n_bootstrap,
                                   gt_points=arguments.gt_points, gt_subsamples=arguments.gt_subsamples,
                                   force_regeneration=arguments.force_regeneration)


if __name__ == "__main__":
    main()
