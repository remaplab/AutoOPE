# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import argparse
import os
import sys
import time

import numpy as np
from numpy.random import RandomState
from obp.dataset import logistic_reward_function, linear_reward_function

from common.constants import LOGS_FOLDER_PATH, EXPERIMENTS_LOGS_FOLDER_NAME
from black_box.data.data_transformation import OutputType
from common.evaluation.evaluation_utils import mk_log_dir, get_pi_e_synthetic_datasets, save_config, \
    save_summarized_results, get_processing_time_file_path, save_processing_time, get_log_folder_path
from common.evaluation.plots import save_es_figures, plot_performance_estimatorwise
from common.ope_estimators import get_op_estimators, get_q_model_by_name, get_estimator_by_name
from common.evaluation.synthetic_data_evaluation import SyntheticDataEvaluation

if os.path.dirname(__file__) == '':
    sys.path.append('../../src')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')



def parse_arges():
    # get argument
    parser = argparse.ArgumentParser()
    # number of actions
    parser.add_argument('--n_actions', type=int, default=10)
    # number of dimensions of context
    parser.add_argument('--dim_context', type=int, default=10)
    # type of reward
    parser.add_argument('--reward_type', type=str, choices=['binary', 'continuous'], default='binary')
    # type of reward function
    parser.add_argument('--reward_function', type=int, choices=[0, 1],
                        help='0:logistic_reward_function, 1:linear_reward_function', default=0)
    reward_function_dict = {0: logistic_reward_function, 1: linear_reward_function}
    # beta of behavior policy 1
    parser.add_argument('--beta_1', type=float, help='beta_1 for conventional method', required=True)
    # beta of behavior policy 2
    parser.add_argument('--beta_2', type=float, help='beta_2 for conventional method', required=True)
    # sample size of log data 1
    parser.add_argument('--n1', type=int, help='n_rounds for beta_1', default=1000)
    # sample size of log data 2
    parser.add_argument('--n2', type=int, help='n_rounds for beta_2', default=1000)
    # stratify
    parser.add_argument('--stratify', type=int, help='Fit q-models stratifying the 3-fold split',
                        default=False)
    # random state
    parser.add_argument('--random_state', type=int, default=1)
    # beta of evaluation policy
    parser.add_argument('--beta_list_for_pi_e', nargs="*", type=float, default=np.arange(-10.0, 11.0).tolist())
    # type of evaluation policy
    # 0:No model,
    # 1:IPWLearner with LogisticRegression,
    # 2:IPWLearner with RandomForest,
    # 3:QLearner with LogisticRegression or RidgeRegression,
    # 4:QLearner with RandomForest
    # if you want to use trained model (1-4), please run model/train_opl_model.py with same n_actions, dim_context,
    # reward_function and random_state before running this code.
    parser.add_argument('--model_list_for_pi_e', nargs="*", type=int, default=0)
    # metric to evaluate the accuracy of estimators
    parser.add_argument('--metric', type=str, choices=['mse', 'mean relative-ee'], default='mse')
    # number of bootstrap in estimator selection procedure
    parser.add_argument('--n_bootstrap', type=int, help='number of bootstrap in estimator selection', default=10)
    # number of data generations of log data
    parser.add_argument('--n_data_generation', type=int, help='number of policy selection phase', default=100)
    # pasif parameters
    parser.add_argument('--pasif_k', nargs="*", type=float, help='k for pasif', default=0.2)
    parser.add_argument('--pasif_regularization_weight', nargs="*", type=float,
                        help='regularization_weight for pasif. -999/-998/-997 means automatic search.', default=-997)
    parser.add_argument('--pasif_batch_size', nargs="*", type=int, help='batch_size for pasif', default=2000)
    parser.add_argument('--pasif_n_epochs', nargs="*", type=int, help='n_epochs for pasif', default=5000)
    parser.add_argument('--pasif_optimizer', nargs="*", type=int, help='optimizer for pasif, 0:SGD, 1:Adam', default=0)
    parser.add_argument('--pasif_lr', nargs="*", type=float, help='learning rate for pasif', default=0.001)
    # number of data generations to calculate the ground truth of estimator selection
    parser.add_argument('--gt_n_sampling', type=int, help='number of sampling to calculate ground truth', default=100)
    # directory to save the results
    parser.add_argument('--save_dir', type=str, help='dir path to save results', default=EXPERIMENTS_LOGS_FOLDER_NAME)
    # loading partial results
    parser.add_argument('--load_gt', action='store_true', default=False)
    parser.add_argument('--load_conv', action='store_true', default=False)
    parser.add_argument('--load_bb', action='store_true', default=False)
    parser.add_argument('--load_pasif', action='store_true', default=False)
    parser.add_argument('--load_slope', action='store_true', default=False)
    parser.add_argument('--load_ocv', action='store_true', default=False)
    # number of jobs
    parser.add_argument('--inner_n_jobs', type=int, required=True)
    parser.add_argument('--outer_n_jobs', type=int, required=True)
    parser.add_argument('--outer_n_jobs_gt', type=int, default=1)
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
    parser.add_argument('--valid_estimators', nargs="*", type=str, default=["IPS", "DR", "DR", "DR"])
    parser.add_argument('--valid_q_models', nargs="*", type=str, default=[None, "RF", "LGBM", "Logistic"])
    parser.add_argument('--train_ratio', type=str, default="theory")
    parser.add_argument('--one_stderr_rule', action='store_true', default=True)
    parser.add_argument('--valid_q_model_kwargs', type=str, default={})
    parser.add_argument('--valid_estimator_kwargs', type=str, default={})
    args = parser.parse_args()

    num_pi_e = len(args.beta_list_for_pi_e)
    args.model_list_for_pi_e = [args.model_list_for_pi_e] * num_pi_e
    args.pasif_k = [args.pasif_k] * num_pi_e
    args.pasif_regularization_weight = [args.pasif_regularization_weight] * num_pi_e
    args.pasif_lr = [args.pasif_lr] * num_pi_e
    args.pasif_batch_size = [args.pasif_batch_size] * num_pi_e
    args.pasif_n_epochs = [args.pasif_n_epochs] * num_pi_e
    args.pasif_optimizer = [args.pasif_optimizer] * num_pi_e
    args.reward_function_dict = reward_function_dict

    args.mkdir = not (args.load_gt or args.load_conv or args.load_bb or args.load_pasif)
    args.folder_name = 'beta_1_' + str(args.beta_1) + '_beta_2_' + str(args.beta_2)

    return args



if __name__ == '__main__':
    args = parse_arges()
    hypara_dict, ope_estimators, slope_estimators, q_models = get_op_estimators('binary')
    pi_e = get_pi_e_synthetic_datasets(args.beta_list_for_pi_e, args.model_list_for_pi_e, args.reward_type)

    # Logs
    log_dir_path = get_log_folder_path(args.save_dir, args.mkdir, args.folder_name)
    time_file_path = get_processing_time_file_path(log_dir_path)

    print("\n\n\n\n ____________________ Evaluate {} ____________________".format("Synthetic (beta_1=" + str(args.beta_1) + ', beta_2=' + str(args.beta_2) + ')'))

    # config
    save_config(dir_name=log_dir_path, args=args, ope_estimators=ope_estimators, hypara_dict=hypara_dict,
                q_models=q_models)

    # set instance for evaluation of selection method
    evaluation_of_selection_method = SyntheticDataEvaluation(
        ope_estimators=ope_estimators,
        q_models=q_models,
        stratify=args.stratify,
        estimator_selection_metrics=args.metric,
        n_actions=args.n_actions,
        dim_context=args.dim_context,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        reward_type=args.reward_type,
        reward_function=args.reward_function_dict[args.reward_function],
        n_rounds_1=args.n1,
        n_rounds_2=args.n2,
        test_ratio=None,
        pi_e=pi_e,
        n_data_generation=args.n_data_generation,
        random_state=args.random_state,
        outer_n_jobs=args.outer_n_jobs,
        inner_n_jobs=args.inner_n_jobs,
        n_bootstrap=args.n_bootstrap,
        log_dir_path=log_dir_path,
        outer_n_jobs_gt=args.outer_n_jobs_gt
    )

    processing_time_list, processing_step_list = [time.time()], []

    evaluation_of_selection_method.set_conventional_method_params(evaluation_data='partial_random')
    evaluation_of_selection_method.set_bb_params(
        model_name=args.bb_model_name, opes_type=args.bb_opes_type, metric_opt=args.bb_metric_opt,
        output_type=getattr(OutputType, args.bb_output_type), use_embeddings=args.bb_use_embeddings,
        metadata_n_points=args.bb_metadata_n_points, metadata_n_boot=args.bb_metadata_n_boot,
        metadata_avg=args.bb_metadata_avg, metadata_rwd_type=args.bb_metadata_rwd_type, custom_folder=args.custom_folder
    )
    evaluation_of_selection_method.set_pasif_method_params_wrapper(args)
    valid_q_models = [get_q_model_by_name(q, kwargs, RandomState(args.random_state)) for q, kwargs in zip(
        args.valid_q_models, args.valid_q_models_kwargs)]
    valid_estimators = [get_estimator_by_name(est, kwargs) for est, kwargs in zip(args.valid_estimators,
                                                                                  args.valid_estimators_kwargs)]
    evaluation_of_selection_method.set_ocv_params(valid_estimators=valid_estimators, valid_q_models=valid_q_models,
                                                  K=args.K, train_ratio=args.train_ratio,
                                                  one_stderr_rule=args.one_stderr_rule)
    evaluation_of_selection_method.set_slope_params(slope_estimators)
    evaluation_of_selection_method.set_const_params(['DM_qmodel_LGBMClassifier', 'DM_qmodel_LogisticRegression',
                                                     'DM_qmodel_RandomForestClassifier',
                                                     'DRTuning_qmodel_LGBMClassifier',
                                                     'DRTuning_qmodel_LogisticRegression',
                                                     'DRTuning_qmodel_RandomForestClassifier',
                                                     # 'DRosTuning_qmodel_LGBMClassifier', 'DRosTuning_qmodel_LogisticRegression',
                                                     # 'DRosTuning_qmodel_RandomForestClassifier',
                                                     'IPWTuning',
                                                     # 'SGDRTuning_qmodel_LGBMClassifier', 'SGDRTuning_qmodel_LogisticRegression',
                                                     # 'SGDRTuning_qmodel_RandomForestClassifier', 'SGIPWTuning',
                                                     # 'SNDR_qmodel_LGBMClassifier', 'SNDR_qmodel_LogisticRegression',
                                                     # 'SNDR_qmodel_RandomForestClassifier',
                                                     'SNIPW',
                                                     # 'SwitchDRTuning_qmodel_LGBMClassifier', 'SwitchDRTuning_qmodel_LogisticRegression',
                                                     # 'SwitchDRTuning_qmodel_RandomForestClassifier'
                                                     ])

    # Ground Truth
    evaluation_of_selection_method.set_ground_truth(args.load_gt, args.gt_n_sampling)
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "Ground-Truth")

    # evaluation of Conventional estimator selection method
    evaluation_of_selection_method.evaluate_conventional_selection_method(args.load_conv)
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
    evaluation_of_selection_method.evaluate_constant()
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "Constant")

    # evaluation of Random estimator selection method
    evaluation_of_selection_method.evaluate_random()
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "Random")

    print('Save result')

    # summarized result
    save_summarized_results(dir_name=log_dir_path, evaluation_of_selection_method=evaluation_of_selection_method)

    # figure
    x_label = r'Evaluation Policy ($\beta_e$)'
    #title = 'Setting 1\n' + r"$(\beta_1, \beta_2) = ({}, {})$".format(args.beta_1, args.beta_2)
    title = 'Logging 1' if args.beta_1 == -2 else 'Logging 2'
    save_es_figures(evaluation_of_selection_method, log_dir_path, pi_e, x_label, title)
    plot_performance_estimatorwise(evaluation_of_selection_method.estimator_selection_gt, pi_e, log_dir_path,
                                   "estimators_gt_mse", title, x_label,
                                   'Ground-Truth MSE', 'mse')
    #plot_performance_estimatorwise(evaluation_of_selection_method.conventional_es_res, pi_e, log_dir_path,
    #                               "conv_estimators_mse", title, x_label,
    #                               'Conventional MSE', 'estimated mse')
    plot_performance_estimatorwise(evaluation_of_selection_method.black_box_es_res, pi_e, log_dir_path,
                                   "bb_estimators_mse", title, x_label,
                                   'AutoOPE MSE', 'estimated mse')
    #plot_performance_estimatorwise(evaluation_of_selection_method.pasif_es_res, pi_e, log_dir_path,
    #                               "pasif_estimators_mse", title, x_label,
    #                               'PAS-IF MSE', 'estimated mse')

    # processing time
    save_processing_time(time_file_path, processing_time_list, processing_step_list, "END")
