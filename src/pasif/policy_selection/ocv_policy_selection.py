from common.policy_selection.base_policy_selection import BasePolicySelection
from pasif.estimator_selection.ocv_estimator_selection import OCVEstimatorSelection



class OCVPolicySelection(BasePolicySelection):
    """
     Off-Policy Cross-Validation policy selection method
    """

    def __init__(self, ope_estimators, q_models, stratify, estimator_selection_metrics='mse', data_type='synthetic',
                 random_state=None, log_dir='./'):
        super().__init__(ope_estimators, q_models, stratify, estimator_selection_metrics, data_type, random_state,
                         log_dir, save=True)
        self.train_valid_ratio = None
        self.valid_q_model_kwargs = None
        self.valid_estimator_kwargs = None
        self.valid_q_model = None
        self.valid_estimator = None
        self.one_standard_error_rule = None
        self.K = None
        self.policy_selection_name = 'ocv'



    def set_ocv_params(self, valid_estimator, valid_q_model=None, K=10, train_ratio='theory', one_stderr_rule=True):
        self.K = K
        self.one_standard_error_rule = one_stderr_rule
        self.valid_estimator = valid_estimator
        self.valid_q_model = valid_q_model
        self.train_valid_ratio = train_ratio
        self.policy_selection_name += "_" + self.valid_estimator.estimator_name
        if self.valid_q_model is not None:
            self.policy_selection_name += '_' + self.valid_q_model.__name__




    def do_estimator_selection(self, pi_e_dist_train, log_data_train, n_bootstrap, policy_name, i_task):
        """
        Perform estimator selection on D^(pre) dataset, to avoid potential bias

        Args:
            pi_e_dist_train: action distribution by the evaluation policy
            log_data_train: logging data
            n_bootstrap: number of bootstrap sampling iterations
            policy_name: name identifier of the evaluation policy
            i_task: iteration index of the outer loop
        """
        # estimator selection
        ocv_es = OCVEstimatorSelection(
            ope_estimators=self.ope_estimators,
            q_models=self.q_models,
            metrics=self.estimator_selection_metrics,
            data_type='real',
            random_state=self.random_state,
            i_task=i_task,
            partial_res_file_name_root=self.get_partial_res_file_path_root() + policy_name + '_',
            stratify=self.stratify
        )
        ocv_es.set_ocv_params(valid_estimator=self.valid_estimator,
                              valid_q_model=self.valid_q_model,
                              K=self.K,
                              train_ratio=self.train_valid_ratio,
                              one_stderr_rule=self.one_standard_error_rule)
        ocv_es.set_real_data(log_data=log_data_train, pi_e_dist=pi_e_dist_train[policy_name])
        ocv_es.evaluate_estimators(n_bootstrap=n_bootstrap, n_jobs=self.inner_n_jobs)
        return ocv_es
