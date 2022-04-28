from math import atan2, pi

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.metrics import loss_functions
from utils import custom

pio.templates.default = "simple_white"


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    ds = np.load(filename)
    X = ds[:, :-1]
    y = ds[:, -1]
    return X, y
    # raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        perp = Perceptron(callback=lambda p, x, i: losses.append(p.loss(X, y)))
        perp.fit(X, y)
        # Plot figure
        fig = go.Figure()
        fig.add_scatter(x=[_ + 1 for _ in range(len(losses))], y=losses)
        fig.update_layout(title= f"{n}",
            xaxis_title=f"Number of Perceptron Iterations",
            yaxis_title="Loss"
        )
        fig.show()
        # raise NotImplementedError()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda.fit(X, y)
        gnb.fit(X, y)
        lda_pred = lda.predict(X)
        gnb_pred = gnb.predict(X)
        y = y.astype(int)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # from IMLearn.metrics import accuracy


        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            rf"$\textbf{{ NGB with accuracy {loss_functions.accuracy(y, gnb_pred)} }}$",
            rf"$\textbf{{ LDA with accuracy {loss_functions.accuracy(y, lda_pred)} }}$"
            ))

        # left graph
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=gnb_pred, symbol=y,
                                             colorscale=[custom[2], custom[4]])), row=1, col=1)
        # left elipses
        for i in range(3):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=1)

        # left markers
        fig.add_trace(
            go.Scatter(x=gnb.mu_.transpose()[0], y=gnb.mu_.transpose()[1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x")), row=1, col=1)

        # right graph
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=lda_pred, symbol=y,
                                             colorscale=[custom[2], custom[4]])), row=1, col=2)
        # right elipses
        for i in range(3):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        # right markers
        fig.add_trace(
            go.Scatter(x=lda.mu_.transpose()[0], y=lda.mu_.transpose()[1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x")), row=1, col=2)

        fig.update_layout(
            title_text=rf"$\textbf{{(1) {f} Dataset}}$")
        # fig.write_image(f"./{f}.png")
        fig.show()
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
