import copy
import csv
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
from obp.dataset import OpenBanditDataset, MultiClassToBanditReduction
from obp.policy import Random, BernoulliTS
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ucimlrepo import fetch_ucirepo

from common.constants import OBD_FOLDER_PATH, CIFAR10_FOLDER_PATH, CLASSIFICATION_DATA_ID_MAP, \
    REAL_DATASETS_FOLDER_PATH, LOGS_FOLDER_PATH



class LearnedBehaviorPolicy:
    """class to use trained policy for SyntheticBanditDataset
    """

    def __init__(self, model):
        """
        Set trained policy model

        Args:
            model (obp.policy): trained policy model
        """
        self.model = model

    def behavior_policy_function_predict_proba(self, context, action_context, random_state):
        """This can be used for behavior policy of SyntheticBanditDataset

        Args:
            context (np.array): context
            action_context (np.array): Conveniently set up for use within the SyntheticBanditDataset.
            random_state (int): Conveniently set up for use within the SyntheticBanditDataset.
        Returns:
            pd.DataFrame, dict: dataframe with columns=[policy_name, estimator_name, estimated_policy_value, rank], and estimator selection result for each evaluation policy
        """
        predicted_action_dist = self.model.predict_score(context)
        predicted_action_dist = predicted_action_dist[:, :, 0]
        return predicted_action_dist


def mk_log_dir(parent_dir, dir_name):
    dir_list = glob.glob(os.path.join(parent_dir, '*'))
    same_dir_count = 0
    for existing_dir in dir_list:
        if dir_name in existing_dir:
            same_dir_count += 1
    if same_dir_count == 0:
        mk_dir_name = os.path.join(parent_dir, dir_name)
    else:
        mk_dir_name = os.path.join(parent_dir, dir_name + '_ver' + str(same_dir_count))
    os.makedirs(mk_dir_name)
    return mk_dir_name


def save_config(dir_name, args, ope_estimators, hypara_dict, q_models):
    file_path = dir_name + '/config.csv'
    with open(file_path, 'w') as f:
        writer = csv.writer(f)

        writer.writerow(['arguments'])
        dict_args = vars(args)
        for arg_key in dict_args:
            writer.writerow([arg_key, dict_args[arg_key]])

        writer.writerow([])
        writer.writerow(['ope_estimators'])
        for estimator in ope_estimators:
            writer.writerow([str(estimator)])

        writer.writerow([])
        writer.writerow(['hypara_space'])
        for hypara_name in hypara_dict.keys():
            writer.writerow([hypara_name, hypara_dict[hypara_name]])

        writer.writerow([])
        writer.writerow(['q_models'])
        for q_model in q_models:
            writer.writerow([str(q_model)])


def save_summarized_results(dir_name, evaluation_of_selection_method):
    file_path = dir_name + '/summarized_results'
    estimator_sel_res = pd.DataFrame()
    policy_sel_res = pd.DataFrame()
    results_list = evaluation_of_selection_method.get_mean_evaluation_results_of_estimator_selection()
    for index, mean_res, std_res in results_list:
        if mean_res is not None:
            for key in mean_res.keys():
                mean_res[key]['relative_regret_mean'] = [mean_res[key].pop('relative_regret')]
                mean_res[key]['rank_correlation_coefficient_mean'] = [mean_res[key].pop('rank_correlation_coefficient')]
                mean_res[key]['mse_mean'] = [mean_res[key].pop('mse')]
                mean_res[key]['relative_regret_std'] = [std_res[key].pop('relative_regret')]
                mean_res[key]['rank_correlation_coefficient_std'] = [std_res[key].pop('rank_correlation_coefficient')]
                mean_res[key]['mse_std'] = [std_res[key].pop('mse')]
                tmp = pd.concat([pd.DataFrame.from_dict({key: mean_res[key]}, orient='index')], names=['policy'])
                tmp = pd.concat({index: tmp}, names=['method'])
                estimator_sel_res = pd.concat([estimator_sel_res, tmp])
    estimator_sel_res.to_csv(file_path + '_estimator_selection.csv')

    results_list = evaluation_of_selection_method.get_mean_evaluation_results_of_policy_selection()
    for index, mean_res, std_res in results_list:
        if mean_res:
            mean_res['relative_regret_mean'] = [mean_res.pop('relative_regret')]
            mean_res['rank_correlation_coefficient_mean'] = [mean_res.pop('rank_correlation_coefficient')]
            mean_res['relative_regret_std'] = [std_res.pop('relative_regret')]
            mean_res['rank_correlation_coefficient_std'] = [std_res.pop('rank_correlation_coefficient')]
            tmp = pd.concat([pd.DataFrame.from_dict({index: mean_res}, orient='index')], names=['method'])
            policy_sel_res = pd.concat([policy_sel_res, tmp])
    policy_sel_res.to_csv(file_path + '_policy_selection.csv')

    # Count how many wins and draws
    all_methods_names = estimator_sel_res.index.get_level_values(0).unique()
    for curr in all_methods_names:
        regret_curr = estimator_sel_res.loc[curr, 'relative_regret_mean']
        wins_draws = pd.DataFrame(data=[[0, 0]], index=['All'], columns=["Wins", "Draws"])
        total_wins, total_draws = regret_curr.copy(), regret_curr.copy()
        total_wins[:], total_draws[:] = True, True

        for baseline in all_methods_names:
            regret_baseline = estimator_sel_res.loc[baseline, 'relative_regret_mean']
            if not baseline == curr:
                wins = regret_curr < regret_baseline
                draws = regret_curr == regret_baseline
                total_wins = np.logical_and(total_wins, wins)
                total_draws = np.logical_and(total_draws, draws)
                wins_draws.loc[baseline, "Wins"] = wins.sum()
                wins_draws.loc[baseline, "Draws"] = draws.sum()
        wins_draws.loc['All', 'Wins'] += total_wins.sum()
        wins_draws.loc['All', 'Draws'] += total_draws.sum()

        wins_draws.to_csv(dir_name + '/' + curr + 'wins_draws_on_mean_rel_regret.csv')

        pasif_present = 'pasif' in estimator_sel_res.index
        bb_results = estimator_sel_res.loc['black_box', 'relative_regret_mean']
        if pasif_present:
            pasif_results = estimator_sel_res.loc['pasif', 'relative_regret_mean']
        wins_draws_bb = pd.DataFrame()
        if pasif_present:
            wins_draws_pasif = pd.DataFrame()
        total_wins = bb_results.copy()
        total_wins[:] = True
        total_draws = bb_results.copy()
        total_draws[:] = True
        wins_draws_bb.loc['All', 'Wins'] = 0
        wins_draws_bb.loc['All', 'Draws'] = 0

        for index, mean_res, std_res in results_list:
            if mean_res:
                baseline_res = estimator_sel_res.loc[index, 'relative_regret_mean']
                if not index == 'black_box':
                    win_bb = bb_results < baseline_res
                    draw_bb = bb_results == baseline_res
                    total_wins = np.logical_and(total_wins, win_bb)
                    total_draws = np.logical_and(total_draws, draw_bb)
                    wins_draws_bb.loc[index, "Wins"] = win_bb.sum()
                    wins_draws_bb.loc[index, "Draws"] = draw_bb.sum()
                if not index == 'pasif' and pasif_present:
                    win_pasif = pasif_results < baseline_res
                    draw_pasif = pasif_results == baseline_res
                    wins_draws_pasif.loc[index, "Wins"] = win_pasif.sum()
                    wins_draws_pasif.loc[index, "Draws"] = draw_pasif.sum()
        wins_draws_bb.loc['All', 'Wins'] += total_wins.sum()
        wins_draws_bb.loc['All', 'Draws'] += total_draws.sum()

        wins_draws_bb.to_csv(dir_name + '/bb_wins_draws_on_mean_rel_regret.csv')
        if pasif_present:
            wins_draws_pasif.to_csv(dir_name + '/pasif_wins_draws_on_mean_rel_regret.csv')





def get_legend_label(method: str):
    if method == 'pasif':
        return 'PAS-IF'
    if method == 'sampled_random':
        return 'Random (Sampled)'
    if method == 'exact_random':
        return 'Random (Expectation)'
    if method == 'black_box':
        return 'AutoOPE'
    if method == 'conventional':
        return 'Conventional'
    if method == 'constant':
        return 'Constant'
    if method == 'slope':
        return 'SLOPE'
    if 'opera' in method:
        return 'OPERA'
    if method.find('ocv') > -1:
        method = method.replace('_', ' ')
        method = method.replace(' RandomForestClassifier', '')#'-RF')
        method = method.replace(' LightGBMClassifier', '')#'-LGBM')
        method = method.replace(' LogisticRegression', '')#'-Logistic')
        method = method.replace('ocv', 'OCV')
        method = method.replace('dr', 'DR')
        method = method.replace('ipw', 'IPS')
    return method


def convert_sec_to_hms(sec):
    hour = int(sec / 3600)
    minute = int((sec - 3600 * hour) / 60)
    new_sec = sec - 3600 * hour - 60 * minute
    return hour, minute, new_sec


def get_pi_e_synthetic_datasets(beta_list_for_pi_e: list[float], model_list_for_pi_e: list[int], reward_type: str):
    pi_e = {}
    assert len(beta_list_for_pi_e) == len(model_list_for_pi_e), \
        'Must be len(beta_list_for_pi_e) == len(model_list_for_pi_e)'
    learned_model_dict = {}
    if os.path.dirname(__file__) == '':
        model_partial_path = 'model/'
    else:
        model_partial_path = os.path.dirname(__file__) + '/model/'
    learned_model_dict['binary'] = [None,
                                    model_partial_path + 'ipw_lr_b.pickle',
                                    model_partial_path + 'ipw_rf_b.pickle',
                                    model_partial_path + 'qlr_lr_b.pickle',
                                    model_partial_path + 'qlr_rf_b.pickle']
    learned_model_dict['continuous'] = [None,
                                        model_partial_path + 'ipw_lr_c.pickle',
                                        model_partial_path + 'ipw_rf_c.pickle',
                                        model_partial_path + 'qlr_rr_c.pickle',
                                        model_partial_path + 'qlr_rf_c.pickle']
    for i_pi_e, model_num_for_pi_e in enumerate(model_list_for_pi_e):
        if model_num_for_pi_e != 0:
            with open(learned_model_dict[reward_type][model_num_for_pi_e], 'rb') as f:
                model_of_pi_e = pickle.load(f)

        if model_num_for_pi_e == 0:
            pi_e['beta_' + str(beta_list_for_pi_e[i_pi_e])] = ('beta', beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 1:
            pi_e['model_ipw_lr_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 2:
            pi_e['model_ipw_rf_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 3:
            if reward_type == 'binary':
                pi_e['model_qlr_lr_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                    = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
            elif reward_type == 'continuous':
                pi_e['model_qlr_rr_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                    = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 4:
            pi_e['model_qlr_rf_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
    return pi_e


def get_pi_e_real_datasets(beta_list_for_pi_e: list[float], model_list_for_pi_e: list[int], reward_type: str):
    pi_e = {}
    assert len(beta_list_for_pi_e) == len(model_list_for_pi_e), \
        'Must be len(beta_list_for_pi_e) == len(model_list_for_pi_e)'
    learned_model_dict = {}
    if os.path.dirname(__file__) == '':
        model_partial_path = 'model/'
    else:
        model_partial_path = os.path.dirname(__file__) + '/model/'
    learned_model_dict['binary'] = [None,
                                    model_partial_path + 'ipw_lr_b.pickle',
                                    model_partial_path + 'ipw_rf_b.pickle',
                                    model_partial_path + 'qlr_lr_b.pickle',
                                    model_partial_path + 'qlr_rf_b.pickle']
    learned_model_dict['continuous'] = [None,
                                        model_partial_path + 'ipw_lr_c.pickle',
                                        model_partial_path + 'ipw_rf_c.pickle',
                                        model_partial_path + 'qlr_rr_c.pickle',
                                        model_partial_path + 'qlr_rf_c.pickle']
    for i_pi_e, model_num_for_pi_e in enumerate(model_list_for_pi_e):
        if model_num_for_pi_e != 0:
            with open(learned_model_dict[reward_type][model_num_for_pi_e], 'rb') as f:
                model_of_pi_e = pickle.load(f)

        if model_num_for_pi_e == 0:
            pi_e['beta_' + str(beta_list_for_pi_e[i_pi_e])] = ('beta', beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 1:
            pi_e['model_ipw_lr_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 2:
            pi_e['model_ipw_rf_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 3:
            if reward_type == 'binary':
                pi_e['model_qlr_lr_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                    = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
            elif reward_type == 'continuous':
                pi_e['model_qlr_rr_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                    = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 4:
            pi_e['model_qlr_rf_beta_' + str(beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0 / beta_list_for_pi_e[i_pi_e])
    return pi_e


def get_classification_data(alpha_b: float, alpha_e_list: list[float],
                            classifier_b: ClassifierMixin, classifier_e: ClassifierMixin,
                            random_state_data: int, dataset_name: str, folder_name: str, eval_size: float):
    if dataset_name == 'cifar10':
        x, y, preprocessor_b = get_cifar10_data()
    else:
        dataset_id = CLASSIFICATION_DATA_ID_MAP[dataset_name]
        x, y, preprocessor_b = get_uci_dataset(dataset_id)

    return transform_classification_data_for_ope(x, y, alpha_b, alpha_e_list, classifier_b, classifier_e, preprocessor_b,
                                                 random_state_data, dataset_name, folder_name, eval_size)


def get_uci_dataset(dataset_id: int):
    # fetch dataset
    dataset = fetch_ucirepo(id=dataset_id)

    # data (as pandas dataframes)
    x = dataset.data.features
    y = dataset.data.targets

    #Remove NaN if any
    x = pd.concat((x, y), axis=1).dropna()
    target_name = y.columns[0]
    y = x[target_name]
    x = x.drop(target_name, axis=1)
    x_info = dataset.variables

    # Preprocessing x
    integers = x_info[np.logical_and(x_info["type"] == "Integer", x_info["role"] == "Feature")]["name"]
    countinuous = x_info[np.logical_and(x_info["type"] == "Continuous", x_info["role"] == "Feature")]["name"]
    integers = [x.columns.get_loc(feature_name) for feature_name in integers.to_numpy()]
    countinuous = [x.columns.get_loc(feature_name) for feature_name in countinuous.to_numpy()]

    transformers = []
    for feature_idx in integers:
        cat_enc = MinMaxScaler()
        one_hot_encoder = Pipeline(steps=[("enc_" + str(feature_idx), cat_enc)])
        transformers.append(("pipe_" + str(feature_idx), one_hot_encoder, [feature_idx]))

    for feature_idx in countinuous:
        cat_enc = MinMaxScaler()
        one_hot_encoder = Pipeline(steps=[("enc_" + str(feature_idx), cat_enc)])
        transformers.append(("pipe_" + str(feature_idx), one_hot_encoder, [feature_idx]))

    preprocessor_b = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # Preprocessing y
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y.values.ravel())

    return x, y, preprocessor_b


def get_obd_data(campaign: str, eval_policy_name: str, model_name: str, random_state: int):
    avail_policies = {'random', 'bts'}
    assert eval_policy_name in avail_policies
    log_policy_name = (avail_policies - {eval_policy_name}).pop()

    # Data Loading and Preprocessing
    log_dataset = OpenBanditDataset(behavior_policy=log_policy_name, campaign=campaign, data_path=OBD_FOLDER_PATH)
    log_bandit_feedback = log_dataset.obtain_batch_bandit_feedback()
    eval_dataset = OpenBanditDataset(behavior_policy=eval_policy_name, campaign=campaign, data_path=OBD_FOLDER_PATH)
    eval_bandit_feedback = eval_dataset.obtain_batch_bandit_feedback()

    policy_rnd = Random(n_actions=log_dataset.n_actions, len_list=log_dataset.len_list, random_state=random_state)
    policy_bts = BernoulliTS(n_actions=log_dataset.n_actions, len_list=log_dataset.len_list, campaign=campaign,
                             random_state=random_state, is_zozotown_prior=True)

    if eval_policy_name == 'random':
        pi_b = policy_bts
        pi_e = policy_rnd
    elif eval_policy_name == 'bts':
        pi_b = policy_rnd
        pi_e = policy_bts
    else:
        pi_b = None
        pi_e = None

    log_bandit_feedback['pi_b'] = pi_b.compute_batch_action_dist(n_rounds=log_bandit_feedback["n_rounds"])
    eval_bandit_feedback['pi_b'] = pi_e.compute_batch_action_dist(n_rounds=eval_bandit_feedback["n_rounds"])
    cf_distribution = pi_e.compute_batch_action_dist(n_rounds=log_bandit_feedback["n_rounds"])

    pi_e_dict, eval_bandit_feedbacks_dict = {}, {}
    pi_e_dict[model_name] = (cf_distribution, eval_policy_name)
    eval_bandit_feedbacks_dict[model_name] = eval_bandit_feedback

    return log_bandit_feedback, pi_e_dict, eval_bandit_feedbacks_dict


def get_cifar10_data():
    x, y = [], []
    list_dir = os.listdir(CIFAR10_FOLDER_PATH)
    for file in list_dir:
        if not os.path.isdir(os.path.join(CIFAR10_FOLDER_PATH, file)) and '.' not in file:
            with open(os.path.join(CIFAR10_FOLDER_PATH, file), 'rb') as f:
                data_batch = pickle.load(f, encoding='bytes')
                x.append(data_batch[b'data'])
                y.append(data_batch[b'labels'])
    x = np.concatenate(x)
    y = np.concatenate(y)

    preprocessor_b = MinMaxScaler()

    return x, y, preprocessor_b


def transform_classification_data_for_ope(x: np.ndarray, y: np.ndarray, alpha_b: float, alpha_e_list: list[float],
                                          classifier_b: ClassifierMixin, classifier_e: ClassifierMixin,
                                          preprocessor_b, random_state_data: int, dataset_name: str,
                                          folder_name: str, eval_size: float):
    if os.path.isdir(folder_name):
        log_bandit_feedback = pickle.load(open(os.path.join(folder_name, 'log_bandit_feedback.pickle'), 'rb'))
        pi_e_dict = pickle.load(open(os.path.join(folder_name, 'pi_e_dict.pickle'), 'rb'))
        cf_bandit_feedbacks = pickle.load(open(os.path.join(folder_name, 'cf_bandit_feedbacks.pickle'), 'rb'))
        pi_e_dict = {key: val for key, val in pi_e_dict.items() if key in ['policy_' + str(k) for k in alpha_e_list]}
        cf_bandit_feedbacks = {key: val for key, val in cf_bandit_feedbacks.items() if key in
                               ['policy_' + str(k) for k in alpha_e_list]}
    else:
        pi_e_dict, cf_bandit_feedbacks = {}, {}
        dataset = MultiClassToBanditReduction(X=x, y=y, base_classifier_b=classifier_b, alpha_b=alpha_b,
                                              dataset_name=dataset_name)
        dataset.split_train_eval(eval_size=eval_size, random_state=random_state_data)
        preprocessor_b.fit(dataset.X_tr)
        dataset.X_tr = preprocessor_b.transform(dataset.X_tr)
        dataset.X_ev = preprocessor_b.transform(dataset.X_ev)

        log_bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=random_state_data)
        log_bandit_feedback['action_context'] = None
        log_bandit_feedback['position'] = np.zeros_like(log_bandit_feedback['action'])

        os.makedirs(folder_name)

        for alpha_e in alpha_e_list:
            model_name = 'policy_' + str(alpha_e)
            pi_e = dataset.obtain_action_dist_by_eval_policy(base_classifier_e=classifier_e, alpha_e=alpha_e)
            mean_reward = dataset.calc_ground_truth_policy_value(pi_e)
            pi_e_dict[model_name] = (pi_e, alpha_e)

            # work-around: the whole evaluation batch bandit feedback is not necessary since we already have the
            # ground-truth. To compute the ground-truth in the RealDataEvaluation class, it is only averaged the
            # 'reward' entry of the cf_bandit_feedback dictionary. So this is sufficient
            cf_bandit_feedback = {'n_rounds': 1, 'reward': np.array([mean_reward])}
            cf_bandit_feedbacks[model_name] = cf_bandit_feedback

        pickle.dump(log_bandit_feedback, open(os.path.join(folder_name, 'log_bandit_feedback.pickle'), 'wb'))
        pickle.dump(pi_e_dict, open(os.path.join(folder_name, 'pi_e_dict.pickle'), 'wb'))
        pickle.dump(cf_bandit_feedbacks, open(os.path.join(folder_name, 'cf_bandit_feedbacks.pickle'), 'wb'))

    return log_bandit_feedback, pi_e_dict, cf_bandit_feedbacks


def save_processing_time(file_path, processing_time_list, processing_step_list, new_step):
    processing_time_list.append(time.time())
    processing_step_list.append(new_step)
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['process', 'hour', 'minute', 'second'])
        for i in range(len(processing_step_list)):
            process_time = convert_sec_to_hms(sec=processing_time_list[i + 1] - processing_time_list[i])
            writer.writerow([processing_step_list[i], process_time[0], process_time[1], process_time[2]])

        process_time = convert_sec_to_hms(sec=processing_time_list[-1] - processing_time_list[0])
        writer.writerow(['total', process_time[0], process_time[1], process_time[2]])


def get_processing_time_file_path(log_dir_path):
    # Prevent to delete older processing time files
    file_path = log_dir_path + '/processing_time.csv'
    version = 0
    while os.path.exists(file_path):
        version += 1
        file_path = log_dir_path + '/processing_time_' + 'ver' + str(version) + '.csv'
    return file_path



def get_real_world_data(dataset, random_state_data, **kwargs):
    if dataset == 'obd':
        model_name = 'policy_' + kwargs['obd_cf_policy']
        log_bandit_feedback, pi_e, cf_bandit_feedback = get_obd_data(campaign=kwargs["obd_campaign"],
                                                                     eval_policy_name=kwargs["obd_cf_policy"],
                                                                     model_name=model_name,
                                                                     random_state=random_state_data)

    else:
        base_classifier_b = LogisticRegression(max_iter=10000, random_state=random_state_data)
        base_classifier_e = LogisticRegression(max_iter=10000, random_state=random_state_data)
        log_bandit_feedback, pi_e, cf_bandit_feedback = get_classification_data(
            kwargs["class_alpha_b"],
            kwargs["class_alpha_e_list"],
            base_classifier_b,
            base_classifier_e,
            random_state_data,
            dataset,
            str(os.path.join(REAL_DATASETS_FOLDER_PATH, dataset, 'bandits_' + str(random_state_data))),
            kwargs["class_eval_size"]
        )
    return log_bandit_feedback, pi_e, cf_bandit_feedback



def get_log_folder_path(save_dir, mkdir, folder_name):
    log_root = os.path.join(LOGS_FOLDER_PATH, save_dir)
    log_dir = os.path.join(log_root, folder_name)
    if mkdir:
        log_dir = mk_log_dir(parent_dir=log_root, dir_name=folder_name)
    return log_dir
