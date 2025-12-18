import glob
import os

import numpy as np
import pandas as pd
from numpy.random import RandomState
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from black_box.common.constants import PLOTS_FOLDER_NAME, DATA_FOLDER_NAME
from black_box.evaluation.plots import plot_feature_against_errors_per_estimator, plot_feature_against_errors, \
    plot_features_dist



def load(working_dir, relative: bool = False, load_embeddings: bool = False) -> (DataFrame, DataFrame, DataFrame,
                                                                                 DataFrame):
    working_dir = os.path.join(working_dir, DATA_FOLDER_NAME)
    common_working_dir = os.path.join(working_dir, '../..')
    features = pd.read_csv(get_file_path(working_dir, "ctx"))
    errors = pd.read_csv(get_file_path(common_working_dir, "errors"))
    gen_params = pd.read_csv(get_file_path(common_working_dir, "synthetic_datasets_params"))
    estimators_features = None
    if load_embeddings:
        estimators_features = pd.read_csv(get_file_path(common_working_dir, "est_embed"))
    if relative:
        target = pd.read_csv(get_file_path(working_dir, "rel_ee"))
    else:
        target = pd.read_csv(get_file_path(working_dir, "se"))
    return features, target, estimators_features, errors, gen_params



def get_file_path(working_dir: str, file_root_name: str):
    try:
        return glob.glob(os.path.join(working_dir, file_root_name + "_*.csv"))[0]
    except IndexError:
        return None



def clean_from_errors(x: DataFrame, y: DataFrame, gen_params: DataFrame, errors: DataFrame, error_type: str) -> (
        DataFrame, DataFrame):
    if error_type == 'all':
        not_errors = errors.isna().all(axis=1)
        x = x[not_errors]
        y = y[not_errors]

    if np.isnan(y).any().any():  # if some NaNs are still present, remove them
        nan_mask = ~np.isnan(y).any(axis=1)
        y = y[nan_mask]
        x = x[nan_mask]
        nan_mask = ~x.isna().any(axis=1)
        y = y[nan_mask]
        x = x[nan_mask]
        #gen_params = gen_params[not_errors]
    return x, y, gen_params



def load_split(test_perc: float, working_dir: str, train_features_plots: bool = False, rng: RandomState = None,
               relative: bool = False, load_embeddings: bool = False, error_type: str = 'all', plot_n_jobs: int = -1,
               plot_features_distributions: bool = True):
    x, y, embed, errors, gen_params = load(relative=relative, load_embeddings=load_embeddings, working_dir=working_dir)
    if 'noavg' in working_dir:
        errors = pd.DataFrame(errors.to_numpy().flatten())
    x, y, gen_params = clean_from_errors(x, y, gen_params, errors, error_type)

    if test_perc > 0.0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_perc, shuffle=True, random_state=rng)
    else:
        x_train, x_test, y_train, y_test = x, None, y, None

    plot_folder_path = os.path.join(working_dir, PLOTS_FOLDER_NAME)

    if train_features_plots:
        print('Features against errors per estimator')
        plot_feature_against_errors_per_estimator(x_train, y_train, plot_folder=plot_folder_path, n_jobs=plot_n_jobs)
        print('Features against errors')
        plot_feature_against_errors(x_train, y_train, plot_folder=plot_folder_path, n_jobs=plot_n_jobs)

    if plot_features_distributions:
        plot_features_dist(plot_folder_path, x_train)

    return x_train, x_test, y_train, y_test, embed, gen_params



def filter_features(x_train, x_test, features_subset='all'):
    policy_dep = ['max_action_prob_log', 'min_action_prob_log', 'max_action_prob_cf', 'min_action_prob_cf', 'max_ps',
                  'self_norm_denor', 'n_clipped_weights', 'total_var_dist', 'pearson_chi_squared_dist',
                  'inner_product_dist', 'chebyshev_dist', 'neyman_chi_squared_dist', 'div', 'canberra', 'k_div_log_cf',
                  'k_div_cf_log', 'jensen_shannon_dist', 'kl_divergence_cf_log', 'kl_divergence_log_cf',
                  'kumar_johnson_dist', 'additive_symmetric_chi_squared_dist', 'euclidian_dist', 'kulczynski_dist',
                  'city_block']  # 24

    policy_indep = ['n_samples', 'n_actions', 'n_def_actions', 'context_dim', 'avg_context_var', 'action_var',
                    'reward_type', 'reward_std', 'reward_mean', 'reward_skew', 'reward_kurtosis']  # 11

    estimator_dep = list(set(x_train.columns) - set(policy_dep) - set(policy_indep))  # 8

    if features_subset == 'all':
        return x_train, x_test
    if features_subset == 'policy_dep':
        return x_train[policy_dep], x_test[policy_dep]
    if features_subset == 'policy_indep':
        return x_train[policy_indep], x_test[policy_indep]
    if features_subset == 'estimator_dep':
        return x_train[estimator_dep], x_test[estimator_dep]
    if features_subset == 'no_kl':
        all_but_kl = list(set(x_train.columns) - {'kl_divergence_cf_log', 'kl_divergence_log_cf'})
        return x_train[all_but_kl], x_test[all_but_kl]



def filter_data(x_train, y_train, data_filter: str = None, rng=None):
    if data_filter == 'actions':
        x_train = x_train[x_train['n_actions'] <= 5]
    elif data_filter == 'KL':
        x_train = x_train[x_train['kl_divergence_log_cf'] <= 0.1]
    elif data_filter is not None and data_filter.isdecimal():  # 15953
        try:
            data_filter = int(data_filter)
        except Exception as e:
            data_filter = float(data_filter)
        x_train, _ = train_test_split(x_train, train_size=data_filter, shuffle=True, random_state=rng)

    print(x_train.shape[0])
    y_train = y_train.loc[x_train.index, :]
    return x_train, y_train
