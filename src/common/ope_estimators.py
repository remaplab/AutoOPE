import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from obp.ope import InverseProbabilityWeightingTuning as IPWTuning, DirectMethod as DM, DoublyRobustTuning as DRTuning, \
    SelfNormalizedInverseProbabilityWeighting as SNIPW, SwitchDoublyRobustTuning as SwitchDRTuning, \
    SelfNormalizedDoublyRobust as SNDR, DoublyRobustWithShrinkageTuning as DRosTuning, \
    SubGaussianInverseProbabilityWeightingTuning as SGIPWTuning, SubGaussianDoublyRobustTuning as SGDRTuning, \
    InverseProbabilityWeighting as IPW, DoublyRobust as DR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from pasif.aggregate_estimators.opera import OPERA



def get_op_estimators(reward_type):
    # set basic info
    ipw_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf]
    dr_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf]
    switchdr_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf]
    dros_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf]  # config of https://arxiv.org/pdf/2108.13703.pdf
    sgipw_lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sgdr_lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    hypara_dict = {
        'ipw_lambda': ipw_lambda_list,
        'dr_lambda': dr_lambda_list,
        'switchdr_lambda': switchdr_lambda_list,
        'dros_lambda': dros_lambda_list,
        'sgipw_lambdas': sgipw_lambda_list,
        'sgdr_lambdas': sgdr_lambda_list
    }

    slope_ope_estimators = [  # SORTED BY VARIANCE
        IPWTuning(lambdas=ipw_lambda_list, tuning_method='slope', estimator_name='IPWTuning'),
        DRTuning(lambdas=dr_lambda_list, tuning_method='slope', estimator_name='DRTuning'),
        DM(estimator_name='DM'),
    ]

    ope_estimators = [
        IPWTuning(lambdas=ipw_lambda_list, tuning_method='slope', estimator_name='IPWTuning'),
        DM(estimator_name='DM'),
        DRTuning(lambdas=dr_lambda_list, tuning_method='slope', estimator_name='DRTuning'),
        SNIPW(estimator_name='SNIPW'),
        SwitchDRTuning(lambdas=switchdr_lambda_list, tuning_method='slope', estimator_name='SwitchDRTuning'),
        SNDR(estimator_name='SNDR'),
        DRosTuning(lambdas=dros_lambda_list, tuning_method='slope', estimator_name='DRosTuning'),
        SGIPWTuning(lambdas=sgipw_lambda_list, tuning_method='slope', estimator_name='SGIPWTuning'),
        SGDRTuning(lambdas=sgdr_lambda_list, tuning_method='slope', estimator_name='SGDRTuning'),
    ]  # Do not change 'estimator_name' above

    q_models = [RandomForestClassifier, LGBMClassifier, LogisticRegression] if reward_type == 'binary' \
        else [RandomForestRegressor, LGBMRegressor, Ridge]

    opera = OPERA(base_estimators=ope_estimators.copy(), q_models=q_models, random_state=0)
    ope_estimators += [opera]

    return hypara_dict, ope_estimators, slope_ope_estimators, q_models



def get_estimator_by_name(valid_estimator, valid_estimator_kwargs):
    if valid_estimator == 'IPS':
        return IPW(**valid_estimator_kwargs)
    if valid_estimator == 'DR':
        return DR(**valid_estimator_kwargs)
    if valid_estimator == 'DM':
        return DM(**valid_estimator_kwargs)
    return None



def get_q_model_by_name(valid_q_model, valid_q_model_kwargs, random_state):
    verbose = 0
    if valid_q_model == "RF":
        return RandomForestClassifier
    if valid_q_model == "LGBM":
        return LGBMClassifier
    if valid_q_model == "Logistic":
        return LogisticRegression
    return None
