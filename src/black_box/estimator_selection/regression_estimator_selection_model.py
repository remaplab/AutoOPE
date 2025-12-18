import os
import pickle

import numpy as np
import pandas as pd
from numpy.random import RandomState
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Integer, Real, Categorical

from black_box.common.constants import OP_ESTIMATOR_COL_NAME, SINGLE_OUTPUT_COL_NAME
from black_box.data.data_transformation import OutputType
from black_box.estimator_selection.supervised_estimator_selection_model import SupervisedEstimatorSelectionModel
from black_box.evaluation.metrics import ALL_METRICS



class RegressionEstimatorSelectionModel(SupervisedEstimatorSelectionModel):
    def __init__(self, model_name: str, working_dir: str, output_type: OutputType, rng: RandomState = None,
                 single_y_for_train: bool = False, model_params_dict: dict = {}, embeddings: DataFrame = None):
        assert output_type in [OutputType.NO_TRANSFORMATION], "Must not be a reward transformation"
        super().__init__(model_name=model_name, output_type=output_type, rng=rng, single_y_for_train=single_y_for_train,
                         model_params_dict=model_params_dict, embeddings=embeddings, working_dir=working_dir)
        self.metrics_to_monitor = ALL_METRICS

    def _fit(self, x_train, y_train):
        if self.embeddings is not None:
            x_train = self._add_estimators_features(x_train)
            x_train = x_train.drop(OP_ESTIMATOR_COL_NAME, axis=1)
        self._build_pipeline(x_train)
        print("Ready to fit...")
        self.pipeline.fit(x_train, y_train)
        print("Fitted!")
        self.features_names = self.pipeline[0].get_feature_names_out()

    def _build_final_estimator(self):
        if 'GP' in self.model_name:
            self.final_estimator = GaussianProcessRegressor(random_state=self.rng, **self.model_params_dict)
        elif 'RF' in self.model_name:
            self.final_estimator = RandomForestRegressor(random_state=self.rng, n_jobs=-1, **self.model_params_dict)
        elif 'EN' in self.model_name:
            self.final_estimator = ElasticNet(random_state=self.rng, precompute=True, **self.model_params_dict)
        else:
            print("Wrong model name:", self.model_name)

        from sklearn.compose import TransformedTargetRegressor
        model = TransformedTargetRegressor(self.final_estimator, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
        self.final_estimator = TransformedTargetRegressor(regressor=model, transformer=MinMaxScaler(), check_inverse=False)

    def get_hyperparmeters_search_space(self, x_train_len: int):
        if 'RF' in self.model_name:
            return [Integer(50, 500, name='n_estimators'),
                    Categorical(["squared_error"], name="criterion"),  # "friedman_mse", "poisson"], name="criterion"),
                    Integer(1, 100, name='max_depth'),
                    Integer(2, 50, name='min_samples_split', prior='uniform'),
                    Integer(1, 50, name='min_samples_leaf', prior='uniform'),
                    # Real(,, name = "min_weight_fraction_leaf")
                    Real(0.1, 1., name='max_features'),
                    # Integer(, , name='max_leaf_nodes'),
                    # Real(, , name="min_impurity_decrease"),
                    # Categorical([True], name="bootstrap"),
                    Categorical([True], name="oob_score"),
                    # Categorical([False], name="warm_start"),
                    # Categorical([True], name="ccp_alpha"),
                    Real(0.01, 1., name='max_samples')]
        elif 'EN' in self.model_name:
            return [Categorical([True, False], name='fit_intercept'),
                    Real(1e-6, 10, prior='log-uniform', name='alpha'),
                    Real(1e-6, 1e-2, prior='log-uniform', name='tol'),
                    Real(0, 1, name='l1_ratio'),
                    Categorical([True, False], name='positive'),
                    Categorical(['cyclic', 'random'], name='selection'),]

    def predict_score(self, x) -> DataFrame:
        pivot = None
        if self.single_y_for_train:
            all_actions_per_x = np.repeat(self._unpivoted_values, x.shape[0])
            x = DataFrame(np.tile(x, (len(self._unpivoted_values), 1)), columns=x.columns,
                          index=np.tile(x.index, len(self._unpivoted_values)))
            x[OP_ESTIMATOR_COL_NAME] = all_actions_per_x
            if self.embeddings is not None:
                x = self._add_estimators_features(x)
                x, pivot = x.drop(OP_ESTIMATOR_COL_NAME, axis=1), x[OP_ESTIMATOR_COL_NAME]
        scores = self.pipeline.predict(x)
        if self.single_y_for_train:  # len(scores.shape) >= 2 and scores.shape[1] <= 1 or len(scores.shape) < 2:
            if self.embeddings is not None:
                x = pd.concat((x, pivot), axis=1)
            x, scores = self._multi_output(x, DataFrame(scores, index=x.index, columns=[SINGLE_OUTPUT_COL_NAME]))
        return scores

    def _multi_output(self, x: DataFrame, y: DataFrame) -> (DataFrame, DataFrame):
        index_col_name = 'index'
        df_x_y = x.copy()
        df_x_y[SINGLE_OUTPUT_COL_NAME] = y.copy()
        df_x_y.reset_index(inplace=True, names=index_col_name)
        excluded = [OP_ESTIMATOR_COL_NAME]
        if self.embeddings is not None:
            excluded = self.embeddings.columns.tolist()
        group_by = [col for col in x.columns if col not in excluded] + [index_col_name]

        # workaround to include nan (pivot_table cannot handle nan here, but can handle -np.inf)
        # First check that no -np.inf are present in df_x_y
        assert not (df_x_y == -np.inf).to_numpy().any()
        df_x_y.fillna(-np.inf, inplace=True)

        df_x_y = df_x_y.pivot_table(index=group_by, columns=OP_ESTIMATOR_COL_NAME, values=SINGLE_OUTPUT_COL_NAME)
        df_x_y = df_x_y.reset_index()
        df_x_y.set_index(index_col_name, inplace=True)

        # Replacing -np.inf with np.nan
        df_x_y.replace(to_replace=-np.inf, value=np.nan)
        return df_x_y.drop(self._unpivoted_values, axis=1), df_x_y[self._unpivoted_values]

    def save_current_model(self):
        pickle.dump(self.pipeline, open(os.path.join(self.METRIC_FOLDER, 'best_pipeline.pkl'), 'wb'))
        pickle.dump(self._unpivoted_values, open(os.path.join(self.METRIC_FOLDER, 'unpivoted_values.pkl'), 'wb'))

    def load_trained_best_model(self, metric_opt: str = None, custom_folder: str = None):
        self.set_optimized_metric_folders_paths(metric_opt, custom_folder)
        self.pipeline = pickle.load(open(os.path.join(self.METRIC_FOLDER, 'best_pipeline.pkl'), 'rb'))
        self._unpivoted_values = pickle.load(open(os.path.join(self.METRIC_FOLDER, 'unpivoted_values.pkl'), 'rb'))
        self.final_estimator = self.pipeline[-1]
        self._restore_folders()
