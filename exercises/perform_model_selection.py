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
    y = lambda x, i: f(x) + noises[i]  # np.random.normal(0, noise)
    xs = np.linspace(-1.2, 2, n_samples)
    noise_data = np.asarray([[x, y(x, ind)] for ind, x in enumerate(xs)])
    trainX, trainY, testX, testY = split_train_test(pd.DataFrame(noise_data[:, 0]),
                                                    pd.Series(noise_data[:, 1]), float(2) / 3)
    trainX = np.squeeze(trainX)
    trainY = np.squeeze(trainY)
    testX = np.squeeze(testX)
    testY = np.squeeze(testY)

    q1 = go.Figure()

    q1.add_scatter(x=xs, y=[f(i) for i in xs], name="noiseless", mode="markers")
    q1.add_scatter(x=testX, y=[y(x, ind) for ind, x in enumerate(testX)], name="test", mode="markers")
    q1.add_scatter(x=trainX, y=[y(x, ind) for ind, x in enumerate(trainX)], name="train", mode="markers")
    q1.update_layout(title=f"Sample Representation With Noise: {noise} and {n_samples} Samples")
    q1.write_image(f"./Q1N{noise}S{n_samples}.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    degs = [i for i in range(11)]
    train_error = [0] * 11
    validation_error = [0] * 11
    for k in degs:
        estimator = PolynomialFitting(k)
        train_error[k], validation_error[k] = cross_validate(estimator,
                                                             trainX,
                                                             trainY,
                                                             mean_square_error)
    q2 = go.Figure()
    q2.add_scatter(x=degs, y=train_error, name="train", mode="markers")
    q2.add_scatter(x=degs, y=validation_error, name="validation", mode="markers")
    q2.update_layout(title=f"Training and Validation Scores Of Models With Noise: {noise} and {n_samples} Samples")
    q2.write_image(f"./Q2N{noise}S{n_samples}.png")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    kmin = np.argmin(validation_error)
    poly = PolynomialFitting(kmin)
    poly.fit(np.asarray(trainX), np.asarray(trainY))
    test_error = np.round(poly.loss(np.asarray(testX), np.asarray(testY)), 2)
    print(kmin, test_error)



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
    X, y = datasets.load_diabetes(return_X_y=True)
    indexes = np.arange(0, y.shape[0], dtype=int)
    np.random.shuffle(indexes)

    trainX = X[indexes[:n_samples], :]
    trainY = y[indexes[:n_samples]]
    testX = X[indexes[n_samples+1:], :]
    testY = y[indexes[n_samples+1:]]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    reg_ridge_vals = np.linspace(0, 1, n_evaluations, )
    reg_lasso_vals = np.linspace(0, 2, n_evaluations)
    lasso_train_err = []
    lasso_valid_err = []
    ridge_train_err = []
    ridge_valid_err = []
    cv = 5

    for lam in reg_lasso_vals:
        lasso_est = Lasso(alpha=lam)
        train_err, valid_err = cross_validate(lasso_est, trainX, trainY, mean_square_error, cv)
        lasso_train_err.append(train_err)
        lasso_valid_err.append(valid_err)

    for lam in reg_ridge_vals:
        ridge_est = RidgeRegression(lam)
        train_err, valid_err = cross_validate(ridge_est, trainX, trainY, mean_square_error, cv)
        ridge_train_err.append(train_err)
        ridge_valid_err.append(valid_err)

    fig = go.Figure()
    fig.add_scatter(x=reg_lasso_vals, y=lasso_train_err, name="lasso train_err", mode="markers")
    fig.add_scatter(x=reg_lasso_vals, y=lasso_valid_err, name="lasso valid_err", mode="markers")
    fig.add_scatter(x=reg_ridge_vals, y=ridge_train_err, name="ridge train_err", mode="markers")
    fig.add_scatter(x=reg_ridge_vals, y=ridge_valid_err, name="ridge valid_err", mode="markers")
    fig.update_layout(title="Different ranges for lamda parameter")
    fig.write_image("./Q7.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    lasso_min_lam = reg_lasso_vals[np.argmin(lasso_valid_err)]
    ridge_min_lam = reg_ridge_vals[np.argmin(ridge_valid_err)]

    #Optimal ridge
    ridge_est = RidgeRegression(lam=ridge_min_lam)
    ridge_est.fit(trainX, trainY)
    print(f"Minimum lamda for ridge: {ridge_min_lam} with loss: {mean_square_error(ridge_est.predict(testX), testY)}")

    #Optimal lasso
    lasso_est = Lasso(alpha=lasso_min_lam)
    lasso_est.fit(trainX, trainY)
    print(f"Minimum lamda for lasso: {lasso_min_lam} with loss: {mean_square_error(lasso_est.predict(testX), testY)}")

    #Lease sqaure
    ls_est = LinearRegression()
    ls_est.fit(trainX, trainY)
    print(f"The Test Error the LS Model yielded equals { mean_square_error(ls_est.predict(testX),testY)}")


if __name__ == '__main__':
    np.random.seed(0)

    # PART I
    select_polynomial_degree(n_samples=100, noise=5)
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    # PART II
    select_regularization_parameter()
