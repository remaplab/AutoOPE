import argparse
from os import makedirs
from os.path import join

import numpy as np

from black_box.common.constants import FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME
from black_box.common.utils import opes_factory, get_metadataset_working_dir
from black_box.data.data_load_utils import load_split, filter_features, filter_data
from black_box.data.data_transformation import OutputType
from black_box.evaluation.metrics import ALL_METRICS


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
    parser.add_argument('--val_perc', type=float, default=None)
    parser.add_argument('--n_calls_opt', type=int, default=50)
    parser.add_argument('--plot_n_jobs', type=int, default=-1)
    parser.add_argument('--random_state_opt', type=int, default=1234)
    parser.add_argument('--random_state_split', type=int, default=5678)
    parser.add_argument('--metadata_n_points', type=int, default=250000)
    parser.add_argument('--metadata_n_boot', type=int, default=10)
    parser.add_argument('--metadata_avg', action='store_true', default=False)
    parser.add_argument('--metadata_rwd_type', type=str, choices=['bin', 'cont', 'mix'], default='bin')
    parser.add_argument('--features_subset', type=str, choices=['all', 'policy_dep', 'policy_indep', 'estimator_dep', 'no_kl'],
                        default='all')
    parser.add_argument('--data_filter', default=None)  # choices=['actions', 'KL', None, float, int]
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
    x_train, y_train = filter_data(x_train, y_train, args.data_filter, rng)
    custom_folder = None
    if args.features_subset not in [None, 'all']:
        if use_embeddings:
            custom_folder = join(FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME, args.features_subset)
        else:
            custom_folder = join(FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME, args.features_subset + "_no-embed")

    if args.data_filter is not None:
        custom_folder = join(FEATURES_SUBSET_EXPERIMENT_FOLDER_NAME, args.data_filter)

    # MODEL TRAINING AND TESTING
    # Initializing a random number generator to have reproducible fitting (even if data are generated)
    opes = opes_factory(est_embed, rng, opes_type, model_name, output_type, working_dir)

    rng = np.random.RandomState(seed=args.random_state_opt)

    opes.optimize(x_train=x_train, y_train=y_train, metric_to_opt=metric_to_opt, n_calls=n_calls_opt, rng=rng,
                  restart_from_checkpoint=restart_from_checkpoint, custom_folder=custom_folder, perc_val=args.val_perc)

    if args.val_perc is None:
        opes.cross_validate_best_model(x_train=x_train, y_train=y_train, metric_opt=metric_to_opt,
                                       custom_folder=custom_folder)

    opes.fit_predict_best_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, plot=plot_res,
                                metric_opt=metric_to_opt, save_res=save, save_model=save, custom_folder=custom_folder)



if __name__ == "__main__":
    main()
