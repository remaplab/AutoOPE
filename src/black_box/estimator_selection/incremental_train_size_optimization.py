import argparse
from os.path import join
import os
from copy import deepcopy

import numpy as np

from black_box.common.constants import MODELS_FOLDER_NAME, TRAIN_SIZE_EXPERIMENT_FOLDER_NAME, \
    FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME
from black_box.common.utils import opes_factory, get_metadataset_working_dir
from black_box.data.data_load_utils import load_split, filter_features, filter_data
from black_box.data.data_transformation import OutputType
from black_box.evaluation.metrics import ALL_METRICS
from black_box.evaluation.plots import plot_train_size_varying_experiment_res


def main():
    # get command-line argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--opes_type', type=str, choices=['regression', 'classification', 'online', 'offline'],
                        default='regression')
    parser.add_argument('--model_name', type=str, default='RF_REG')
    parser.add_argument('--output_type', type=str, choices=['NO_TRANSFORMATION', 'ERROR_RWD', 'TOP1', 'RANK_RWD'],
                        default='NO_TRANSFORMATION')
    parser.add_argument('--metric_to_opt', type=str, choices=ALL_METRICS, default='REGRET')
    parser.add_argument('--plot_res', action="store_true", default=False)
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--err_type', type=str, choices=['all'], default='all')
    parser.add_argument('--single_y_for_train', action='store_true', default=True)
    parser.add_argument('--train_features_plots', action='store_true', default=False)
    parser.add_argument('--restart_from_checkpoint', action='store_true', default=True)
    parser.add_argument('--use_embeddings', action='store_true', default=True)
    parser.add_argument('--test_perc', type=float, default=0.2)
    parser.add_argument('--n_calls_opt', type=int, default=100)
    parser.add_argument('--plot_n_jobs', type=int, default=-1)
    parser.add_argument('--random_state_opt', type=int, default=1234)
    parser.add_argument('--random_state_split', type=int, default=5678)
    parser.add_argument('--min_data_size', type=int, default=100)
    parser.add_argument('--max_data_size', type=int, default=None)
    parser.add_argument('--n_different_data_size', type=int, default=10)
    parser.add_argument('--metadata_n_points', type=int, default=250000)
    parser.add_argument('--metadata_n_boot', type=int, default=10)
    parser.add_argument('--metadata_avg', action='store_true', default=True)
    parser.add_argument('--metadata_rwd_type', type=str, choices=['bin', 'cont', 'mix'], default='bin')
    parser.add_argument('--data_filter', type=str, choices=['actions', 'KL', None], default=None)
    parser.add_argument('--features_subset', type=str, choices=[
        'all', 'policy_dep', 'policy_indep', 'estimator_dep', 'no_kl'], default='all')
    parser.add_argument('--force_train', action='store_true', default=False)
    args = parser.parse_args()

    opes_type = args.opes_type
    model_name = args.model_name
    output_type = getattr(OutputType, args.output_type)
    metric_to_opt = args.metric_to_opt
    plot_res = args.plot_res
    save = args.save
    restart_from_checkpoint = args.restart_from_checkpoint
    use_embeddings = args.use_embeddings
    test_perc = args.test_perc
    n_calls_opt = args.n_calls_opt
    n_different_data_size = args.n_different_data_size
    min_data_size = args.min_data_size
    max_data = args.max_data_size

    working_dir = get_metadataset_working_dir(args.metadata_rwd_type, args.metadata_n_points, args.metadata_n_boot,
                                              args.metadata_avg)

    # DATA LOADING
    # Initializing a random number generator
    rng = np.random.RandomState(seed=args.random_state_split)
    x_train, x_test, y_train, y_test, est_embed, _ = load_split(test_perc=test_perc, rng=rng,
                                                                train_features_plots=args.train_features_plots,
                                                                load_embeddings=use_embeddings,
                                                                error_type=args.err_type,
                                                                plot_n_jobs=args.plot_n_jobs,
                                                                working_dir=working_dir)

    x_train, x_test = filter_features(x_train, x_test, args.features_subset)
    x_train, y_train = filter_data(x_train, y_train, args.data_filter)
    custom_folder = ""
    if args.features_subset not in [None, 'all']:
        if use_embeddings:
            custom_folder = join(FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME, args.features_subset)
        else:
            custom_folder = join(FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME, args.features_subset + "_no-embed")

    if args.data_filter is not None:
        custom_folder = join(FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME, args.data_filter)

    if max_data is None:
        max_data = x_train.shape[0]
    log_space = np.logspace(np.log10(min_data_size), np.log10(max_data), num=n_different_data_size)
    log_space = [int(size) for size in log_space]
    print('Train size space', log_space)
    res_gp_list, res_train_list, res_test_list = [], [], []

    experiment_folder = join(working_dir, MODELS_FOLDER_NAME, output_type.name, model_name, metric_to_opt,
                             custom_folder if not custom_folder == "" else TRAIN_SIZE_EXPERIMENT_FOLDER_NAME)
    os.makedirs(experiment_folder, exist_ok=True)

    for data_size in log_space:
        print("Train size", data_size)
        subsample_idx = rng.choice(np.array(x_train.index), size=data_size, replace=False)
        i_folder = os.path.join(custom_folder, str(data_size) + '_samples')
        x_train_sub = x_train.loc[subsample_idx, :]
        y_train_sub = y_train.loc[subsample_idx, :]

        # MODEL TRAINING AND TESTING
        # Initializing a random number generator to have reproducible fitting (even if data are generated)
        rng_copy = deepcopy(rng)
        opes = opes_factory(est_embed, rng_copy, opes_type, model_name, output_type, working_dir)

        # Optimization
        rng = np.random.RandomState(seed=args.random_state_opt)
        #res_gp = opes.optimize(x_train=x_train_sub, y_train=y_train_sub, metric_to_opt=metric_to_opt,
        #                       n_calls=n_calls_opt, rng=rng,
        #                       restart_from_checkpoint=restart_from_checkpoint,
        #                       custom_folder=i_folder)
        # Clear objective
        #res_gp['specs']['args']['func'] = None
        #res_gp_list.append(res_gp)

        # Performance Best Model
        res_train, res_test = opes.load_res(metric_to_opt, i_folder)
        if res_train is None or res_test is None or args.force_train:
            res_train, res_test = opes.fit_predict_best_model(x_train=x_train_sub, y_train=y_train_sub, x_test=x_test,
                                                              y_test=y_test, plot=plot_res, metric_opt=metric_to_opt,
                                                              save_res=save, save_model=save, custom_folder=i_folder)
        res_train_list.append(res_train)
        res_test_list.append(res_test)
    plot_train_size_varying_experiment_res(res_gp_list, res_train_list, res_test_list, log_space, metric_to_opt,
                                           experiment_folder)


if __name__ == "__main__":
    main()
