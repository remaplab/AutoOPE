from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer

from black_box.data.data_transformation import OutputType
from black_box.estimator_selection.base_estimator_selection_model import BaseEstimatorSelectionModel
from black_box.estimator_selection.clipping_transformer import ClippingTransformer



class SupervisedEstimatorSelectionModel(BaseEstimatorSelectionModel, metaclass=ABCMeta):
    THRESHOLD_NUM_FEATURES = 1e10



    def __init__(self, model_name: str, working_dir: str, output_type: OutputType, rng: RandomState = None,
                 single_y_for_train: bool = False, model_params_dict: dict = {}, embeddings: DataFrame = None):
        super().__init__(model_name=model_name, output_type=output_type, rng=rng, single_y_for_train=single_y_for_train,
                         model_params_dict=model_params_dict, embeddings=embeddings, working_dir=working_dir)
        self.final_estimator = None
        self.high_model_score_better = self.high_true_score_better
        self.features_names = None



    @abstractmethod
    def _fit(self, x_train, y_train):
        pass



    def predict_best(self, x: DataFrame) -> Series:
        rank_pos = self.predict_rank_pos(x, higher_score_is_better=self.high_model_score_better)
        return self._get_best_from_rank_pos(rank_pos)



    @abstractmethod
    def predict_score(self, x: DataFrame) -> DataFrame:
        pass



    def _build_preprocessing_pipeline(self, x_train: DataFrame):
        return self._build_preprocessing_pipeline_skewed(x_train)



    def _build_preprocessing_pipeline_skewed(self, x_train: DataFrame):
        transformers = []
        for cat_feature in self.cat_features:
            categories = x_train[cat_feature].unique()
            #categories = categories[categories.astype(str) != str(np.nan)]
            drop = [np.nan] if (categories.astype(str) == str(np.nan)).any() else None
            categories = 'auto'#[categories.tolist()] if len(categories) > 1 else 'auto'
            cat_enc = OneHotEncoder(categories=categories, handle_unknown="error", drop=drop, sparse=False)
            #one_hot_encoder = Pipeline(steps=[("enc_" + cat_feature, cat_enc)], verbose=10)
            transformers.append(("1hot_" + cat_feature, cat_enc, [cat_feature]))

        pos_skewed_features = self.get_skewed_features(x_train)
        negative_features = x_train[self.num_features].columns[(x_train[self.num_features] < 0).any()].tolist()
        pos_not_skewed_features = list(set(self.num_features) - set(pos_skewed_features) - set(negative_features))
        assert sorted(pos_skewed_features + pos_not_skewed_features + negative_features) == sorted(self.num_features)

        clipping_transformer = ClippingTransformer(clip_max=self.THRESHOLD_NUM_FEATURES)
        #nan_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, feature_names_out='one-to-one',
                                              check_inverse=False)
        scaler = MinMaxScaler()
        neg_scaler = MinMaxScaler(feature_range=(-1, 1))

        pos_not_skewed_num_features_pipe = Pipeline(steps=[("clipping_pns", deepcopy(clipping_transformer)),
                                                           #("imp_nan", nan_imputer),
                                                           ('scaler_pns', deepcopy(scaler))],
                                                    verbose=10)

        neg_num_features_pipe = Pipeline(steps=[("clipping_neg", deepcopy(clipping_transformer)),
                                                # ("imp_nan", nan_imputer),
                                                ('scaler_neg', neg_scaler)],
                                         verbose=10)

        skewed_num_features_pipe = Pipeline(steps=[("clipping_sk", deepcopy(clipping_transformer)),
                                                   ("log_transformer_sk", log_transformer),
                                                   ('scaler_sk', deepcopy(scaler))],
                                            verbose=10)

        transformers.append(("pos_not_skewed_scaler", pos_not_skewed_num_features_pipe, pos_not_skewed_features))
        transformers.append(("neg_scaler", neg_num_features_pipe, negative_features))
        transformers.append(("skewed_transformer", skewed_num_features_pipe, pos_skewed_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            #n_jobs=len(transformers),
            verbose=10
        )



    def _build_preprocessing_pipeline_simple(self, x_train: DataFrame):
        transformers = []
        for cat_feature in self.cat_features:
            categories = x_train[cat_feature].unique()
            #categories = categories[categories.astype(str) != str(np.nan)]
            drop = [np.nan] if (categories.astype(str) == str(np.nan)).any() else None
            categories = 'auto'#[categories.tolist()] if len(categories) > 1 else 'auto'
            cat_enc = OneHotEncoder(categories=categories, handle_unknown="error", drop=drop, sparse=False)
            #one_hot_encoder = Pipeline(steps=[("enc_" + cat_feature, cat_enc)], verbose=10)
            transformers.append(("1hot_" + cat_feature, cat_enc, [cat_feature]))

        clipping_transformer = ClippingTransformer(clip_max=self.THRESHOLD_NUM_FEATURES)
        clipping_pipe = Pipeline(steps=[("clipping", clipping_transformer)], verbose=10)

        transformers.append(("clipping_pipe", clipping_pipe, self.num_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            #n_jobs=len(transformers),
            verbose=10
        )



    @abstractmethod
    def _build_final_estimator(self):
        pass



    def _build_pipeline(self, x_train: DataFrame):
        self._inspect_features_types(x_train)
        self._build_preprocessing_pipeline(x_train)
        self._build_final_estimator()
        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor),
                   ("model", self.final_estimator)],
            verbose=10,
            #memory='cache'
        )



    @abstractmethod
    def get_hyperparmeters_search_space(self, x_train_len: int):
        pass



    @abstractmethod
    def save_current_model(self):
        pass



    @abstractmethod
    def load_trained_best_model(self, metric_opt: str = None, custom_folder: str = None):
        pass



    def get_skewed_features(self, x_train):
        x = x_train[self.num_features].copy()
        x[x > self.THRESHOLD_NUM_FEATURES] = self.THRESHOLD_NUM_FEATURES
        skew_vals = x.skew()
        skewed_features = np.array(self.num_features)[np.abs(skew_vals) > 1]
        pos_skewed_features = skewed_features[(x[skewed_features] >= 0).all(axis=0)]
        #neg_skewed_features = list(set(skewed_features) - set(pos_skewed_features))
        return pos_skewed_features.tolist()  #, neg_skewed_features
