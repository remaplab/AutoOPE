from dataclasses import dataclass
from typing import Optional

import numpy as np
from obp.ope import RegressionModel
from obp.utils import check_bandit_feedback_inputs
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import check_random_state, check_scalar



@dataclass
class RegressionModelStratified(RegressionModel):
    stratify: bool = True

    def fit_predict(
            self,
            context: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            pscore: Optional[np.ndarray] = None,
            position: Optional[np.ndarray] = None,
            action_dist: Optional[np.ndarray] = None,
            n_folds: int = 1,
            random_state: Optional[int] = None
    ) -> np.ndarray:
        """Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
        One fold is used if it is not possible to provide at least one sample of each class to each fold.

        Note
        ------
        When `n_folds` is larger than 1, the cross-fitting procedure is applied.
        See the reference for the details about the cross-fitting technique.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities (propensity score) of a behavior policy
            in the training set of logged bandit data.
            If None, the the behavior policy is assumed to be uniform random.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a regression model assumes that only a single action is chosen for each data.
            When `len_list` > 1, an array must be given as `position`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.
            When either 'iw' or 'mrdr' is set to `fitting_method`, `action_dist` must be given.

        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the regression model is trained on the whole logged bandit data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.

        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards of new data estimated by the regression model.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context)
        else:
            q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        if self.stratify:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        else:
            print("Not Stratified!")
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        try:
            for train_idx, test_idx in kf.split(context, reward):
                action_dist_tr = (
                    action_dist[train_idx] if action_dist is not None else action_dist
                )
                self.fit(
                    context=context[train_idx],
                    action=action[train_idx],
                    reward=reward[train_idx],
                    pscore=pscore[train_idx],
                    position=position[train_idx],
                    action_dist=action_dist_tr,
                )
                q_hat[test_idx, :, :] = self.predict(context=context[test_idx])
        except (ValueError, IndexError) as e:
            print("Exception thrown: ", e)
            print("Probably one data class has less samples than the number of folds. Trying only one fold.")
            try:
                self.fit(
                    context=context,
                    action=action,
                    reward=reward,
                    pscore=pscore,
                    position=position,
                    action_dist=action_dist,
                )
                return self.predict(context=context)
            except IndexError as e:
                if position.max() > 0:
                    print("Exception thrown: ", e)
                    print("Probably not all classes for each position. Assume len_list=1")
                    original_len_list = self.len_list
                    self.len_list = 1
                    self.fit(
                        context=context,
                        action=action,
                        reward=reward,
                        pscore=pscore,
                        position=np.zeros_like(position),
                        action_dist=action_dist,
                    )
                    predicted_rwd = self.predict(context=context)
                    predicted_rwd = np.concatenate([predicted_rwd] * original_len_list, axis=-1)
                    return predicted_rwd
        return q_hat
