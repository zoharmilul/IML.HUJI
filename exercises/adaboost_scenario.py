import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import IMLearn.metrics.loss_functions as lf


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)


    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)
    losses = [ada.partial_loss(test_X, test_y, i + 1) for i in range(n_learners)]
    errors = [ada.partial_loss(train_X, train_y, i + 1) for i in range(n_learners)]
    x_ax = [i+1 for i in range(n_learners)]
    fig = go.Figure()
    fig.update_layout(title=f"Q1 with noise: {noise}")
    fig.add_scatter(x=x_ax, y=losses, name="test error")
    fig.add_scatter(x=x_ax, y=errors, name="train error")
    # fig.show()
    fig.write_image(f"./Q1_noise{noise}.png")
    # Question 2: Plotting decision surfaces

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])


    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{Decision Boundaries Of - {t} Learners}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    # fig2.suptitle(f"Q2 with noise {noise}")
    fig2.update_layout(title= f"Q2 with noise {noise}")
    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda X: ada.partial_predict(X, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y,  colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    # fig2.show()
    fig2.write_image(f"./Q2_noise{noise}.png")

    # Question 3: Decision surface of best performing ensemble
    best_pref_idx = np.argmin(losses)
    fig3 = go.Figure()
    fig3.update_layout(title=f"Q3 with noise {noise}\n Decision Boundaries Of - {best_pref_idx + 1} Learners"
                             f"Accuracy: {lf.accuracy(ada.partial_predict(test_X, best_pref_idx), test_y)}")
    fig3.add_traces([decision_surface(lambda X: ada.partial_predict(X, best_pref_idx), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers", showlegend=False,
                                   marker=dict(color=test_y,  colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))])
    # fig3.show()
    fig3.write_image(f"./Q3_noise{noise}.png")


    # Question 4: Decision surface with weighted samples
    D = ada.D_
    D = (D/np.max(D))*5
    boundry = decision_surface(lambda X: ada.predict(X), lims[0], lims[1], showscale=False)
    fig4 = go.Figure()
    fig4.update_layout(title=f"Q4 with noise {noise}")
    fig4.add_traces([boundry, go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                                 marker=dict(size=D, opacity=0.9, color=train_y, colorscale=class_colors(3)))])
    # fig4.show()
    fig4.write_image(f"./Q4_noise{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    for i in [0, 0.4]:
        fit_and_evaluate_adaboost(i)
