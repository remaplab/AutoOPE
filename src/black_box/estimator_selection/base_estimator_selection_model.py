import os
import pickle
from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from numpy.random import RandomState
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, ShuffleSplit
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver, TimerCallback
from skopt.utils import use_named_args, dump, load

from black_box.common.constants import OP_ESTIMATOR_COL_NAME, PLOTS_FOLDER_NAME, RESULTS_FOLDER_NAME, \
    SINGLE_OUTPUT_COL_NAME, MODELS_FOLDER_NAME
from black_box.common.utils import get_rank_pos_from_score, get_rank_from_score, sort_index_and_columns, \
    get_best_from_rank
from black_box.data.data_transformation import transform_output, OutputType
from black_box.evaluation.metrics import ALL_METRICS_FUN_DICT, REGRET_METRICS, ALL_METRICS
from black_box.evaluation.metrics import ERROR_METRICS, RANKING_METRICS, CLASSIFICATION_METRICS
from black_box.evaluation.plots import plot_regression_results, plot_classification_results, plot_ranking_results

pd.set_option('display.max_columns', None)



class BaseEstimatorSelectionModel(BaseEstimator, metaclass=ABCMeta):

    def __init__(self, model_name: str, output_type: OutputType, working_dir: str, rng: RandomState = None,
                 single_y_for_train: bool = None, model_params_dict: dict = {}, embeddings: DataFrame = None):
        self.model_name = model_name
        self.output_type = output_type
        self.high_true_score_better = output_type.high_score_better
        self.rng = rng
        self.single_y_for_train = single_y_for_train
        self.model_params_dict = model_params_dict
        self.embeddings = embeddings
        self.working_dir = working_dir

        self.MODEL_FOLDER = os.path.join(os.path.join(os.path.join(self.working_dir, MODELS_FOLDER_NAME),
                                                      self.output_type.name), self.model_name)
        self.RESULTS_FOLDER = os.path.join(self.MODEL_FOLDER, RESULTS_FOLDER_NAME)
        self.PLOTS_FOLDER = os.path.join(self.MODEL_FOLDER, PLOTS_FOLDER_NAME)
        self.METRIC_FOLDER = None
        self.metrics_to_monitor = None
        self.cat_features = None
        self.num_features = None
        self.plot_regression = False
        self.plot_classification = False
        self.plot_ranking = False
        self._unpivoted_values = None
        self.n_jobs = -1
        self.high_model_score_better = None



    @abstractmethod
    def _fit(self, x_train: DataFrame, y_train: DataFrame):
        pass



    def fit(self, x_train: DataFrame, y_train: DataFrame):
        y_train = transform_output(y_train.copy(), self.output_type)
        if self.single_y_for_train:
            x_train, y_train = self._single_output(x=x_train.copy(), y=y_train)
        self._fit(x_train, y_train)



    def fit_predict(self, x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame,
                    plot: bool = False, save: bool = False) -> (np.ndarray, DataFrame):
        print("\nFitting model ...")
        self.fit(x_train, y_train)
        print('\nTraining evaluation:')
        self.evaluate_model(x_train, y_train, "train_", plot, save)
        print('\nTest evaluation:')
        return self.evaluate_model(x_test, y_test, "test_", plot, save)



    @abstractmethod
    def predict_score(self, x: DataFrame) -> DataFrame:
        pass



    @abstractmethod
    def predict_best(self, x: DataFrame) -> Series:
        pass



    def predict_rank_pos(self, x: DataFrame, higher_score_is_better: bool) -> DataFrame:
        scores = self.predict_score(x)
        return get_rank_pos_from_score(scores, higher_score_is_better)



    def predict_rank(self, x: DataFrame, higher_score_is_better: bool) -> DataFrame:
        scores = self.predict_score(x)
        return get_rank_from_score(scores, higher_score_is_better)



    def _get_best_from_rank_pos(self, rank_pos: DataFrame):
        best = np.tile(rank_pos.columns, (rank_pos.shape[0], 1))
        mask = rank_pos.to_numpy() == np.min(rank_pos.to_numpy())
        return Series(best[mask], name=SINGLE_OUTPUT_COL_NAME, index=rank_pos.index)



    def _add_estimators_features(self, x: DataFrame):
        x = pd.merge(x.reset_index(), self.embeddings, how='left', on=OP_ESTIMATOR_COL_NAME).set_index('index')
        return x



    def _inspect_features_types(self, x):
        self.cat_features = list(x.select_dtypes(include='object', exclude=np.number).columns)
        self.num_features = list(x.select_dtypes(include=np.number, exclude='object').columns)



    def evaluate_model(self, x_pred: DataFrame, y_true: DataFrame, file_prefix: str = "test_",
                       plot: bool = False, save: bool = False) -> DataFrame:
        results, ys, ys_pred = {}, {}, {}
        for metric in self.metrics_to_monitor:
            ys[metric], ys_pred[metric] = self.prepare_metric(x_pred=x_pred, y=y_true, metric=metric, save=save,
                                                              file_prefix=file_prefix)
            results[metric] = [ALL_METRICS_FUN_DICT[metric](ys[metric], ys_pred[metric])]
        df_result = DataFrame.from_dict(results)
        if save:
            self.save_results_as_csv(df_result, file_prefix + "results")
        print(df_result, '\n')
        self.results_plots(df_result, file_prefix, ys, ys_pred, plot)
        return df_result



    def results_plots(self, df_result, file_prefix, ys, ys_pred, plot):
        # Error metrics
        if plot:
            os.makedirs(self.PLOTS_FOLDER, exist_ok=True)
            reg_metrics = [m for m in self.metrics_to_monitor if m in ERROR_METRICS]
            if reg_metrics:  # equivalent to 'if  == []'
                plot_regression_results(ys[reg_metrics[0]].to_numpy(), ys_pred[reg_metrics[0]].to_numpy(),
                                        self.model_name, '\n'.join([r"{}$={:.4f}$".format(metric_name,
                                                                                          df_result[metric_name][0]) for
                                                                    metric_name in list(df_result.columns)]),
                                        model_folder=self.PLOTS_FOLDER,
                                        file_prefix=file_prefix + "regression_")

            # Ranking metrics
            rank_metrics = [m for m in self.metrics_to_monitor if m in RANKING_METRICS]
            if rank_metrics:  # equivalent to 'if  == []'
                plot_ranking_results(ys[rank_metrics[0]].to_numpy(), ys_pred[rank_metrics[0]].to_numpy(),
                                     model_folder=self.PLOTS_FOLDER,
                                     file_prefix=file_prefix + "ranking_")

            # Classification metrics
            class_metrics = [m for m in self.metrics_to_monitor if m in CLASSIFICATION_METRICS]
            if class_metrics:  # equivalent to 'if  == []'
                plot_classification_results(ys[class_metrics[0]].to_numpy(), ys_pred[class_metrics[0]].to_numpy(),
                                            model_folder=self.PLOTS_FOLDER,
                                            file_prefix=file_prefix + "classification_")



    def _single_output(self, x: DataFrame, y: DataFrame) -> (DataFrame, Series):
        data = x.join(y)
        df_unpivoted = data.melt(id_vars=x.columns, value_vars=y.columns, value_name=SINGLE_OUTPUT_COL_NAME,
                                 var_name=OP_ESTIMATOR_COL_NAME, ignore_index=False)
        self._unpivoted_values = np.unique(df_unpivoted[OP_ESTIMATOR_COL_NAME])
        df_unpivoted = df_unpivoted.sample(frac=1., random_state=self.rng, ignore_index=False)
        return df_unpivoted.drop([SINGLE_OUTPUT_COL_NAME], axis=1), df_unpivoted[SINGLE_OUTPUT_COL_NAME]



    def save_results_as_csv(self, to_save, file_name):
        os.makedirs(self.RESULTS_FOLDER, exist_ok=True)
        to_save = DataFrame(to_save)
        to_save.to_csv(os.path.join(self.RESULTS_FOLDER, file_name + ".csv"))



    def fit_predict_best_model(self, x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame,
                               plot: bool = False, metric_opt: str = None, save_res: bool = True,
                               save_model: bool = True, custom_folder: str = None):
        """
        Model fitted only after optimization, then is quicker to load it
        """
        print("Fitting best model")
        assert metric_opt is not None
        self._fit_best_model(x_train, y_train, metric_opt, save_model, custom_folder)
        print('\nTraining evaluation:')
        res_train = self.evaluate_model(x_train, y_train, "train_", plot, save_res)
        print('\nTest evaluation:')
        res_test = self.evaluate_model(x_test, y_test, "test_", plot, save_res)
        self._restore_folders()
        return res_train, res_test



    def cross_validate_best_model(self, x_train: DataFrame, y_train: DataFrame, cv=5, metric_opt: str = None,
                                  custom_folder: str = None):
        print("Cross-validating best model")
        assert metric_opt is not None
        self.set_optimized_metric_folders_paths(metric_opt, custom_folder)
        # Read dictionary pkl file
        with open(os.path.join(self.METRIC_FOLDER, 'best_hyperparams.pkl'), 'rb') as fp:
            params = pickle.load(fp)
            print("Best hyperparameters:", params)

        self._set_params(params)

        monitor_metrics = ALL_METRICS.copy()
        monitor_metrics.remove(metric_opt)
        scoring = {metric_to_compute: make_scorer(metric_to_compute) for metric_to_compute in monitor_metrics}
        res = cross_validate(self, x_train, y_train, cv=cv, n_jobs=-1, scoring=scoring,
                             error_score='raise', return_train_score=True, verbose=10)

        pickle.dump(res, open(os.path.join(
            self.METRIC_FOLDER, "{}-cross_val_metrics_best_model.pickle".format(cv)), 'wb'))

        print(res)

        self._restore_folders()



    def load_res(self, metric, custom_folder: str = None):
        """
        Load results of train and test phase
        """
        self.set_optimized_metric_folders_paths(metric, custom_folder)
        res_train, res_test = None, None
        if os.path.exists(os.path.join(self.RESULTS_FOLDER, "test_results.csv")):
            res_train = pd.read_csv(os.path.join(self.RESULTS_FOLDER, "train_results.csv"), index_col=0)
            res_test = pd.read_csv(os.path.join(self.RESULTS_FOLDER, "test_results.csv"), index_col=0)
        self._restore_folders()
        return res_train, res_test



    def _fit_best_model(self, x_train: DataFrame, y_train: DataFrame, metric_opt: str = None, save_model: bool = True,
                        custom_folder: str = None):
        """
        Model fitted only after optimization, then is quicker to load it
        """
        assert metric_opt is not None
        self.set_optimized_metric_folders_paths(metric_opt, custom_folder)
        # Read dictionary pkl file
        with open(os.path.join(self.METRIC_FOLDER, 'best_hyperparams.pkl'), 'rb') as fp:
            params = pickle.load(fp)
            print("Best hyperparameters:", params)

        self._set_params(params)
        self.fit(x_train, y_train)
        if save_model:
            print('Save best model')
            self.save_current_model()



    @abstractmethod
    def get_hyperparmeters_search_space(self, x_train_len: int):
        pass



    def _set_params(self, params):
        self.model_params_dict = params



    def optimize(self, x_train: DataFrame, y_train: DataFrame, metric_to_opt: str, perc_val = None, n_calls: int = 100,
                 rng: RandomState = None, restart_from_checkpoint: bool = False, cv: int = 5,
                 custom_folder: str = None) -> (Dict, list[float]):
        self.set_optimized_metric_folders_paths(metric_to_opt, custom_folder)
        space = self.get_hyperparmeters_search_space(int(x_train.shape[0] / cv))
        train_gp_res = []



        @use_named_args(space)
        def objective(**params):
            print("\n")
            print('Hyperparameters', params)
            self._set_params(params)
            scoring = {metric_to_opt: make_scorer(metric_to_opt)}

            cv_ = cv
            cv_jobs = cv
            if perc_val is not None:
                cv_ = ShuffleSplit(n_splits=1, test_size=perc_val, random_state=1)
                cv_jobs = -1

            res = cross_validate(self, x_train, y_train, cv=cv_, n_jobs=cv_jobs, scoring=scoring, error_score='raise',
                                 return_train_score=True, verbose=10)

            train_res = res['train_' + metric_to_opt]
            val_res = res['test_' + metric_to_opt]
            train_res_mean = np.mean(train_res)
            val_res_mean = np.mean(val_res)
            train_gp_res.append(train_res_mean)
            pickle.dump(train_gp_res, open(os.path.join(self.METRIC_FOLDER, 'train_gp_res.pickle'), 'wb'))
            pickle.dump(rng, open(os.path.join(self.METRIC_FOLDER, 'rng.pkl'), 'wb'))
            print('Train result:', train_res)
            print('Validation result:', val_res)
            if perc_val is None:
                print('Train results Mean:', train_res_mean)
                print('Validation results Mean:', val_res_mean)
            return val_res_mean



        n_initial_points = int(n_calls * 0.3)
        x0, y0 = None, None

        if restart_from_checkpoint:
            if os.path.exists(os.path.join(self.METRIC_FOLDER, "checkpoint.pkl")):
                print('Restart from checkpoint...')
                rng = pickle.load(open(os.path.join(self.METRIC_FOLDER, 'rng.pkl'), 'rb'))
                train_gp_res = pickle.load(open(os.path.join(self.METRIC_FOLDER, 'train_gp_res.pickle'), 'rb'))
                partial_res = load(os.path.join(self.METRIC_FOLDER, "checkpoint.pkl"))
                x0 = partial_res.x_iters
                y0 = partial_res.func_vals
                already_evaluated_points = len(y0)
                n_calls = n_calls - already_evaluated_points
                n_initial_points = max(n_initial_points - len(x0), 0)
                print('x0:\n', "\n".join([str(x0_ii) for x0_ii in x0]))
                print('y0', y0)
                print('Already evaluated points:', already_evaluated_points)
                print('N° calls:', n_calls)
                if n_calls < 0:
                    n_calls = 0
                    print('New N° calls:', n_calls)
                print('N° initial points:', n_initial_points)
            else:
                print('Checkpoint does not exist. Starting new optimization.')

        checkpoint = CheckpointSaver(os.path.join(self.METRIC_FOLDER, "checkpoint.pkl"), store_objective=False)
        timer = TimerCallback()
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_initial_points,
                             initial_point_generator="random", acq_func="gp_hedge", acq_optimizer="auto", x0=x0,
                             y0=y0, random_state=rng, verbose=10, callback=[checkpoint, timer], n_points=10000,
                             n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise="gaussian", n_jobs=self.n_jobs,
                             model_queue_size=None)



        @use_named_args(space)
        def save_best_hyperparams_as_dict(**params):
            # save dictionary file
            with open(os.path.join(self.METRIC_FOLDER, 'best_hyperparams.pkl'), 'wb') as fp:
                pickle.dump(params, fp)
                print("Best hyperparameters:", params)



        print("Best score=%.4f" % res_gp.fun)
        save_best_hyperparams_as_dict(res_gp['x'])
        dump(res_gp, os.path.join(self.METRIC_FOLDER, 'optimization.pkl'), store_objective=False)
        self._restore_folders()
        res_gp['train_fun'] = train_gp_res[res_gp['x_iters'].index(res_gp['x'])]
        return res_gp



    @abstractmethod
    def save_current_model(self):
        pass



    @abstractmethod
    def load_trained_best_model(self, metric_opt: str = None, custom_folder: str = None):
        pass



    def set_optimized_metric_folders_paths(self, metric, custom_folder=None):
        self.METRIC_FOLDER = os.path.join(self.MODEL_FOLDER, metric)
        if custom_folder is not None:
            self.METRIC_FOLDER = os.path.join(self.METRIC_FOLDER, custom_folder)
        self.RESULTS_FOLDER = os.path.join(self.METRIC_FOLDER, RESULTS_FOLDER_NAME)
        self.PLOTS_FOLDER = os.path.join(self.METRIC_FOLDER, PLOTS_FOLDER_NAME)
        os.makedirs(self.RESULTS_FOLDER, exist_ok=True)
        os.makedirs(self.PLOTS_FOLDER, exist_ok=True)



    def _prepare_error_metric(self, x_pred: DataFrame, y: DataFrame, save: bool = False,
                              file_prefix: str = ""):
        y = transform_output(y, self.output_type)
        y_pred = self.predict_score(x=x_pred)
        y = sort_index_and_columns(y)
        y_pred = sort_index_and_columns(y_pred)
        if save:
            self.save_results_as_csv(y_pred, file_prefix + "predictions")
            self.save_results_as_csv(y, file_prefix + "true_values")
        return y, y_pred



    def _prepare_regret_metric(self, x_pred: DataFrame, y: DataFrame, save: bool = False,
                               file_prefix: str = ""):
        y_pred = self.predict_rank_pos(x=x_pred, higher_score_is_better=self.high_model_score_better)
        y_pred = sort_index_and_columns(y_pred)
        y = sort_index_and_columns(y)
        if save:
            self.save_results_as_csv(y_pred, file_prefix + "predictions")
            self.save_results_as_csv(y, file_prefix + "true_values")
        return y, y_pred



    def _prepare_ranking_metric(self, x_pred: DataFrame, y: DataFrame, save: bool = False,
                                file_prefix: str = ""):
        y = transform_output(y, self.output_type)
        y_pred = self.predict_rank_pos(x=x_pred, higher_score_is_better=self.high_model_score_better)
        y = get_rank_pos_from_score(score=y, higher_score_is_better=self.high_true_score_better)
        y_pred = sort_index_and_columns(y_pred)
        y = sort_index_and_columns(y)
        if save:
            self.save_results_as_csv(y_pred, file_prefix + "predictions")
            self.save_results_as_csv(y, file_prefix + "true_values")
        return y, y_pred



    def _prepare_classification_metric(self, x_pred: DataFrame, y: DataFrame, save: bool = False,
                                       file_prefix: str = ""):
        y = transform_output(y, self.output_type)
        y_pred = self.predict_best(x=x_pred)
        y = get_rank_from_score(score=y, higher_score_is_better=self.high_true_score_better)
        y = sort_index_and_columns(y)
        y = get_best_from_rank(rank=y)
        y_pred = sort_index_and_columns(y_pred)
        y = sort_index_and_columns(y)
        if save:
            self.save_results_as_csv(y_pred, file_prefix + "predictions")
            self.save_results_as_csv(y, file_prefix + "true_values")
        return y, y_pred



    def prepare_metric(self, x_pred: DataFrame, y: DataFrame, metric: str, save: bool = False, file_prefix: str = ""):
        if metric in ERROR_METRICS:
            return self._prepare_error_metric(x_pred=x_pred, y=y, save=save, file_prefix=file_prefix + "regression_")
        if metric in REGRET_METRICS:
            return self._prepare_regret_metric(x_pred=x_pred, y=y, save=save, file_prefix=file_prefix + "regret_")
        elif metric in RANKING_METRICS:
            return self._prepare_ranking_metric(x_pred=x_pred, y=y, save=save, file_prefix=file_prefix + "ranking_")
        elif metric in CLASSIFICATION_METRICS:
            return self._prepare_classification_metric(x_pred=x_pred, y=y, save=save,
                                                       file_prefix=file_prefix + "classification_")



    def _restore_folders(self):
        self.METRIC_FOLDER = None
        self.RESULTS_FOLDER = os.path.join(self.MODEL_FOLDER, RESULTS_FOLDER_NAME)
        self.PLOTS_FOLDER = os.path.join(self.MODEL_FOLDER, PLOTS_FOLDER_NAME)



def make_scorer(metric: str):
    def score(opes_: BaseEstimatorSelectionModel, X: DataFrame, y: DataFrame) -> float:
        y, y_pred = opes_.prepare_metric(x_pred=X, y=y, metric=metric)
        if metric not in ERROR_METRICS + REGRET_METRICS:
            return -ALL_METRICS_FUN_DICT[metric](y, y_pred)
        return ALL_METRICS_FUN_DICT[metric](y, y_pred)



    return score
