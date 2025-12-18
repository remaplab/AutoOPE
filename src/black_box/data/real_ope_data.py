from typing import Dict, List

import numpy as np
from obp.types import BanditFeedback

from black_box.data.base_ope_data import BaseOffPolicyContextBanditData
from common.data.counterfactual_pi_b import get_counterfactual_pscore


class RealOffPolicyContextBanditData(BaseOffPolicyContextBanditData):
    def __init__(self):
        self.cf_data = None
        self.cf_pi_b = None
        self.log_data = None

    def feature_engineering(self, log_data: BanditFeedback, cf_pi_b: np.ndarray) -> Dict[str, List]:
        self.log_data = log_data
        self.cf_pi_b = cf_pi_b

        pscore = get_counterfactual_pscore(self.cf_pi_b, self.log_data['action'], self.log_data['position'])
        self.cf_data = {'pi_b': self.cf_pi_b,
                        'pscore': pscore}

        return self._feature_engineering(log_data=self.log_data, cf_data=self.cf_data)

