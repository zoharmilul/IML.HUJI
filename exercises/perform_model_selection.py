from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """


    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    noises = np.random.normal(0, noise, n_samples)
    y = lambda x, i: f(x) + noises[i] #np.random.normal(0, noise)
    xs = np.linspace(-1.2, 2, n_samples)
    noise_data = np.asarray([[x, y(x, ind)] for ind, x in enumerate(xs)])
    trainX, trainY, testX, testY = split_train_test(pd.DataFrame(noise_data[:, 0]),
                                                    pd.Series(noise_data[:, 1]), float(2)/3)
    trainX = np.squeeze(trainX)
    trainY = np.squeeze(trainY)
    testX = np.squeeze(testX)
    testY = np.squeeze(testY)

    q1 = go.Figure()
    q1.add_scatter(x=xs, y=[f(i) for i in xs], name="noiseless", mode="markers")
    q1.add_scatter(x=testX, y=[y(x, ind) for ind, x in enumerate(testX)], name="test", mode="markers")
    q1.add_scatter(x=trainX, y=[y(x, ind) for ind, x in enumerate(trainX)], name="train", mode="markers")
    q1.write_image(f"./Q1N{noise}S{n_samples}.png")


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    degs = [i for i in range(11)]
    train_error = [0]*11
    validation_error = [0]*11
    for k in degs:
        estimator = PolynomialFitting(k)
        train_error[k], validation_error[k] = cross_validate(estimator,
                                                             trainX,
                                                             trainY,
                                                             mean_square_error)
    q2 = go.Figure()
    q2.add_scatter(x=degs, y=train_error, name="train", mode="markers")
    q2.add_scatter(x=degs, y=validation_error, name="validation", mode="markers")
    q2.write_image(f"./Q2N{noise}S{n_samples}.png")


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    kmin = np.argmin(validation_error)
    poly = PolynomialFitting(kmin)
    poly.fit(np.asarray(trainX), np.asarray(trainY))
    test_error = np.round(poly.loss(np.asarray(testX), np.asarray(testY)), 2)
    print(kmin, test_error, validation_error[kmin])



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500)
    raise NotImplementedError()
