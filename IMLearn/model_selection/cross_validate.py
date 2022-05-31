from __future__ import annotations

import random
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score_arr = []
    validation_score_arr = []

    X = np.asarray(X)
    y = np.asarray(y)
    indexes = np.arange(0, y.shape[0], dtype=int)
    np.random.shuffle(indexes)
    folds = np.array_split(indexes, cv)
    train_err = 0
    valid_err = 0
    for k in folds:

        trainX = np.delete(X, k, 0)
        trainY = np.delete(y, k, 0)

        testX = X[k]
        testY = y[k]

        estimator.fit(trainX, trainY)

        y_pred_train = np.squeeze(estimator.predict(trainX))
        y_pred_test = np.squeeze(estimator.predict(testX))

        train_err += float(scoring(y_pred_train, trainY))
        valid_err += float(scoring(y_pred_test, testY))
        # train_score_arr.append(float(scoring(y_pred_train, trainY)))
        # validation_score_arr.append(float(scoring(y_pred_test, testY)))

    # return float(np.mean(train_score_arr)), float(np.mean(validation_score_arr))
    return train_err/cv, valid_err/cv
