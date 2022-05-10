from __future__ import annotations

import math
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.sign_ = 1
        losses = []
        vals = []
        for i in range(len(X[0])):
            temp = self._find_threshold(X[:, i], y, self.sign_)
            vals.append(temp[0])
            losses.append(temp[1])
        self.j_ = np.argmin(losses)
        self.threshold_ = vals[self.j_]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        response = []
        for val in X[:self.j_]:
            if val >= self.threshold_:
                response.append(self.sign_)
            else:
                response.append(-1 * self.sign_)

        return np.asarray(response)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        min_loss = math.inf
        thresh = 0
        pred = lambda arr, th: [sign if arr[i] >= th else -sign for i in range(len(arr))]
        loss = lambda y_pred, y_true: np.sum([np.abs(y_true[i]) if np.sign(y_true[i]) == np.sign(y_pred[i])
                                              else 0 for i in range(len(y_true))])

        for val in values:
            temp_thresh = val
            temp_pred = pred(values, temp_thresh)
            temp_loss = loss(temp_pred, labels)
            if temp_loss < min_loss:
                min_loss = temp_loss
                thresh = val

        return thresh, min_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        loss = np.sum([np.abs(y[i]) if np.sign(y[i]) == np.sign(y_pred[i])
                       else 0 for i in range(len(y))])
        return float(loss)
