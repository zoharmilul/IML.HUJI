from __future__ import annotations

import math
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
import IMLearn.metrics.loss_functions as lf
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
        losses = []
        vals = []
        signs = []
        for i in range(len(X[0])):
            temp1 = self._find_threshold(X[:, i], y, 1)
            temp2 = self._find_threshold(X[:, i], y, -1)
            if temp2[1] < temp1[1]:
                vals.append(temp2[0])
                losses.append(temp2[1])
                signs.append(-1)
            else:
                vals.append(temp1[0])
                losses.append(temp1[1])
                signs.append(1)

        self.j_ = np.argmin(losses)
        self.threshold_ = vals[self.j_]
        self.sign_ = signs[self.j_]

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

        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

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

        p = np.argsort(values)
        values = values[p]
        labels = labels[p]
        curr_loss = self.wieghted_loss(np.ones(len(labels))*sign, labels)
        loss = curr_loss
        thresh_ind = 0

        for i in range(len(values)):
            if i > 0:
                if np.sign(labels[i-1]) == sign:
                    curr_loss += np.abs(labels[i-1])
                else:
                    curr_loss -= np.abs(labels[i-1])
            if curr_loss < loss:
                loss = curr_loss
                thresh_ind = i

        return values[thresh_ind], loss/np.sum(np.abs(labels))

    def wieghted_loss(self, y_pred, y_true):
        loss = 0
        for i, val in enumerate(y_true):
            if np.sign(val) != y_pred[i]:
                loss += np.abs(val)
        return loss

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
        loss = lf.misclassification_error(y, y_pred)
        return float(loss)
