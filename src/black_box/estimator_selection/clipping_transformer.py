import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array


class ClippingTransformer(TransformerMixin, BaseEstimator):
    """
    Given an interval, values outside the interval are clipped to the interval edges.
    """

    def __init__(self, clip_max: float, clip_min: float = -np.inf):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X, force_all_finite=False)
        return np.clip(X, self.clip_min, self.clip_max)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        return input_features
