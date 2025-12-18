import argparse
import time

from numpy.random import RandomState

from black_box.data.data_transformation import OutputType
from common.constants import EXPERIMENTS_LOGS_FOLDER_NAME, ALL_REAL_WORLD_DATASETS, OBD_POLICIES, OBD_CAMPAIGNS
from common.evaluation.evaluation_utils import save_config, save_summarized_results, save_processing_time, \
    get_processing_time_file_path, \
    get_real_world_data, get_log_folder_path
from common.evaluation.plots import save_es_figures, plot_performance_estimatorwise
from common.evaluation.real_data_evaluation import RealDataEvaluation
from common.ope_estimators import get_op_estimators, get_estimator_by_name, get_q_model_by_name



def parse_args():
    # get argument
    parser = argparse.ArgumentParser()
    # subsampling percentage
    parser.add_argument('--subsampling_ratio', type=float, help='subsampling percentage', required=True)
    # random states
    parser.add_argument('--random_state', type=int, default=1234)
    parser.add_argument('--random_state_data', type=int, default=5678)
    # metric to evaluate the accuracy of estimators
    parser.add_argument('--metric', type=str, choices=['mse', 'mean relative-ee'], default='mse')
    # stratify
    parser.add_argument('--stratify', type=int, help='Fit q-models without stratifying the 3-fold split',
                        default=True)
    # number of bootstrap in estimator selection procedure
    parser.add_argument('--n_bootstrap', type=int, help='number of bootstrap in estimator selection', default=0)
    # number of data generations of log data
    parser.add_argument('--n_data_generation', type=int, help='number of policy selection phase', required=True)
    # directory to save the results
    parser.add_argument('--save_dir', type=str, help='dir path to save results', default=EXPERIMENTS_LOGS_FOLDER_NAME)
    # number of concurrent jobs
    parser.add_argument('--inner_n_jobs', type=int, default=5)
    parser.add_argument('--outer_n_jobs', type=int, required=True)
    # load partial results
    parser.add_argument('--load_gt', action='store_true', default=False)
    parser.add_argument('--load_conv', action='store_true', default=False)
    parser.add_argument('--load_bb', action='store_true', default=False)
    parser.add_argument('--load_pasif', action='store_true', default=False)
    parser.add_argument('--load_slope', action='store_true', default=False)
    parser.add_argument('--load_ocv', action='store_true', default=False)
    parser.add_argument('--outer_n_jobs_gt', type=int, default=1)
    # pasif parameters
    parser.add_argument('--pasif_k', nargs="*", type=float, help='k for pasif', default=0.2)
    parser.add_argument('--pasif_regularization_weight', nargs="*", type=float,
                        help='regularization_weight for pasif. -999/-998/-997 means automatic search.', default=-997)
    parser.add_argument('--pasif_batch_size', nargs="*", type=int, help='batch_size for pasif', default=None)
    parser.add_argument('--pasif_n_epochs', nargs="*", type=int, help='n_epochs for pasif', default=5000)
    parser.add_argument('--pasif_optimizer', nargs="*", type=int, help='pasif optimizer, 0:SGD, 1:Adam', default=0)
    parser.add_argument('--pasif_lr', nargs="*", type=float, help='learning rate for pasif', default=0.001)
    # bb params
    parser.add_argument('--bb_model_name', type=str, default='RF_REG')
    parser.add_argument('--bb_opes_type', type=str, default='regression')
    parser.add_argument('--bb_output_type', type=str, default='NO_TRANSFORMATION')
    parser.add_argument('--bb_metric_opt', type=str, default='REGRET')
    parser.add_argument('--bb_use_embeddings', action='store_true', default=True)
    parser.add_argument('--bb_metadata_n_points', type=int, default=250000)
    parser.add_argument('--bb_metadata_n_boot', type=int, default=10)
    parser.add_argument('--bb_metadata_avg', action='store_true', default=False)
    parser.add_argument('--bb_metadata_rwd_type', type=str, choices=['bin', 'cont', 'mix'], default='bin')
    parser.add_argument('--custom_folder', type=str, default=None)
    # ocv params
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--valid_estimators', nargs="*", type=str, default=["IPS", "DR"])  #, "DR", "DR"])
    parser.add_argument('--valid_q_models', nargs="*", type=str, default=[None, "Logistic"])  # "RF", "LGBM"])
    parser.add_argument('--train_ratio', type=str, default="theory")
    parser.add_argument('--one_stderr_rule', action='store_true', default=True)
    parser.add_argument('--valid_q_models_kwargs', nargs="*", type=str, default=[{}, {}, {}, {}])
    parser.add_argument('--valid_estimators_kwargs', nargs="*", type=str, default=[{}, {}, {}, {}])
    # Real dataset
    parser.add_argument('--dataset', type=str, choices=ALL_REAL_WORLD_DATASETS, required=True)
    # Open Bandit
    parser.add_argument('--obd_campaign', type=str, choices=OBD_CAMPAIGNS, default='all')
    parser.add_argument('--obd_cf_policy', type=str, choices=OBD_POLICIES, default='bts')
    # Classification data
    parser.add_argument('--class_alpha_b', type=float, default=0.2)
    parser.add_argument('--class_alpha_e_list', nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 0.99])
    parser.add_argument('--class_eval_size', type=float, help='data percentage used as evaluation data', default=0.5)

    args = parser.parse_args()

    args.x_label = ""
    if args.dataset == "obd":
        behaviour_policy_name = 'random' if args.obd_cf_policy == 'bts' else 'bts'
        args.folder_name = args.dataset + '_' + behaviour_policy_name
        num_pi_e = 1
        args.dataset_kwargs = {"obd_campaign": args.obd_campaign, "obd_cf_policy": args.obd_cf_policy}
    else:
        num_pi_e = len(args.class_alpha_e_list)
        args.x_label = r'Evaluation Policy ($\alpha_e$)'
        args.folder_name = args.dataset + '_alpha_' + str(args.class_alpha_b)
        args.dataset_kwargs = {"class_alpha_b": args.class_alpha_b, "class_alpha_e_list": args.class_alpha_e_list,
                               "class_eval_size": args.class_eval_size}

    args.pasif_k = [args.pasif_k] * num_pi_e
    args.pasif_regularization_weight = [args.pasif_regularization_weight] * num_pi_e
    args.pasif_lr = [args.pasif_lr] * num_pi_e
    args.pasif_batch_size = [args.pasif_batch_size] * num_pi_e
    args.pasif_n_epochs = [args.pasif_n_epochs] * num_pi_e
    args.pasif_optimizer = [args.pasif_optimizer] * num_pi_e

    args.mkdir = not (args.load_gt or args.load_conv or args.load_bb or args.load_pasif)

    return args



if __name__ == '__main__':
    args = parse_args()
    hypara_dict, ope_estimators, slope_estimators, q_models = get_op_estimators('binary')
    log_bandit_feedback, pi_e, cf_bandit_feedback = get_real_world_data(args.dataset, args.random_state_data,
                                                                        **args.dataset_kwargs)

    # Logs
    log_dir_path = get_log_folder_path(args.save_dir, args.mkdir, args.folder_name)
    time_file_path = get_processing_time_file_path(log_dir_path)

    print("\n\n\n\n ____________________ Evaluate {} ____________________".format(args.dataset))

    # config
    save_config(dir_name=log_dir_path, args=args, ope_estimators=ope_estimators, hypara_dict=hypara_dict,
                q_models=q_models)

    # set instance for evaluation of selection method
    evaluation_of_selection_method = RealDataEvaluation(
        ope_estimators=ope_estimators, q_models=q_models,
        stratify=args.stratify,
        log_dir_path=log_dir_path,
        log_bandit_feedback=log_bandit_feedback,
        eval_bandit_feedback=cf_bandit_feedback, pi_e=pi_e,
        test_ratio=0.5, n_data_generation=args.n_data_generation,
        random_state=args.random_state,
        estimator_selection_metrics=args.metric,
        outer_n_jobs=args.outer_n_jobs,
        inner_n_jobs=args.inner_n_jobs,
        n_bootstrap=args.n_bootstrap,
        undersampling_ratio=args.subsampling_ratio,
        outer_n_jobs_gt=args.outer_n_jobs_gt
    )

    processing_time_list, processing_step_list = [time.time()], []

    # Set params
    evaluation_of_selection_method.set_conventional_method_params(evaluation_data='partial_random')
    evaluation_of_selection_method.set_bb_params(
        model_name=args.bb_model_name, opes_type=args.bb_opes_type, metric_opt=args.bb_metric_opt,
        output_type=getattr(OutputType, args.bb_output_type), use_embeddings=args.bb_use_embeddings,
        metadata_n_points=args.bb_metadata_n_points, metadata_n_boot=args.bb_metadata_n_boot,
        metadata_avg=args.bb_metadata_avg, metadata_rwd_type=args.bb_metadata_rwd_type, custom_folder=args.custom_folder
    )
    evaluation_of_selection_method.set_pasif_method_params_wrapper(args, list(pi_e.keys()))
    valid_q_models = [get_q_model_by_name(q, kwargs, RandomState(args.random_state)) for q, kwargs in zip(
        args.valid_q_models, args.valid_q_models_kwargs)]
    valid_estimators = [get_estimator_by_name(est, kwargs) for est, kwargs in zip(args.valid_estimators,
                                                                                  args.valid_estimators_kwargs)]
    evaluation_of_selection_method.set_ocv_params(valid_estimators=valid_estimators, valid_q_models=valid_q_models,
                                                  K=args.K, train_ratio=args.train_ratio,
                                                  one_stderr_rule=args.one_stderr_rule)
    evaluation_of_selection_method.set_slope_params(slope_estimators)
    evaluation_of_selection_method.set_const_params([#'DM_qmodel_LGBMClassifier',
                                                     #'DM_qmodel_LogisticRegression',
                                                     #'DM_qmodel_RandomForestClassifier',
                                                     #'DRTuning_qmodel_LGBMClassifier',
                                                     #'DRTuning_qmodel_LogisticRegression',
                                                     #'DRTuning_qmodel_RandomForestClassifier',
                                                     # 'DRosTuning_qmodel_LGBMClassifier', 'DRosTuning_qmodel_LogisticRegression',
                                                     # 'DRosTuning_qmodel_RandomForestClassifier',
                                                     #'IPWTuning',
                                                     # 'SGDRTuning_qmodel_LGBMClassifier', 'SGDRTuning_qmodel_LogisticRegression',
                                                     # 'SGDRTuning_qmodel_RandomForestClassifier', 'SGIPWTuning',
                                                     # 'SNDR_qmodel_LGBMClassifier', 'SNDR_qmodel_LogisticRegression',
                                                     # 'SNDR_qmodel_RandomForestClassifier',
                                                     # 'SNIPW',
                                                     # 'SwitchDRTuning_qmodel_LGBMClassifier', 'SwitchDRTuning_qmodel_LogisticRegression',
                                                     # 'SwitchDRTuning_qmodel_RandomForestClassifier',
                                                     'opera'])

    # Ground Truth
    evaluation_of_selection_method.set_ground_truth(args.load_gt)
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "Ground-Truth")

    # evaluation of Conventional estimator selection method
    #evaluation_of_selection_method.evaluate_conventional_selection_method(args.load_conv)
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "Conventional")

    # evaluation of PAS-IF estimator selection method
    evaluation_of_selection_method.evaluate_pasif(args.load_pasif)
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "PAS-IF")

    # evaluation of SLOPE estimator selection method
    evaluation_of_selection_method.evaluate_slope(args.load_slope)
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "SLOPE")

    # evaluation of OCV estimator selection method
    evaluation_of_selection_method.evaluate_ocv(args.load_ocv)
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "OCV")

    # evaluation of AutoOPE estimator selection method
    evaluation_of_selection_method.evaluate_bb_selection_method(args.load_bb)
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "AutoOPE")

    # evaluation of Constant estimator selection method
    #evaluation_of_selection_method.evaluate_constant()
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "Constant")

    # evaluation of Random estimator selection method
    #evaluation_of_selection_method.evaluate_random()
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "Random")

    print('Save Result')

    # summarized result
    save_summarized_results(dir_name=log_dir_path, evaluation_of_selection_method=evaluation_of_selection_method)

    # figures
    save_es_figures(evaluation_of_selection_method, log_dir_path, pi_e, args.x_label, args.dataset)
    plot_performance_estimatorwise(evaluation_of_selection_method.estimator_selection_gt, pi_e, log_dir_path,
                                   "estimators_gt_mse", args.dataset, args.x_label,
                                   'Ground-Truth MSE', 'mse')
    plot_performance_estimatorwise(evaluation_of_selection_method.black_box_es_res, pi_e, log_dir_path,
                                   "bb_estimators_mse", args.dataset, args.x_label,
                                   'AutoOPE MSE', 'estimated mse')
    #plot_performance_estimatorwise(evaluation_of_selection_method.pasif_es_res, pi_e, log_dir_path,
    #                               "pasif_estimators_mse", args.dataset, args.x_label,
    #                               'PAS-IF MSE', 'estimated mse')

    # processing time
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "END")
