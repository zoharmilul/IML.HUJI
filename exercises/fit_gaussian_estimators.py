from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    true_exp = 10
    true_var = 1
    samples = np.random.normal(true_exp, true_var, 1000)
    uni_gauss = UnivariateGaussian().fit(samples)
    print(uni_gauss.mu_, uni_gauss.var_)

    # raise NotImplementedError()

    # Question 2 - Empirically showing sample mean is consistent
    # chunks_array = np.ndarray((1,100))
    # for i in range(100):
    #     chunks_array[i] = samples[:(i+1) * 10+1].mean
    chunck_array = [samples[:(i+1) * 10].mean for i in range(100)]
    fig = make_subplots(rows = 100,cols = 1).add_traces[go.Scatter(x=[i*10 for i in range(1,101)], y=chunck_array,
                                                                   mode='dots', marker=dict(color="black"), showlegend=False)]

    raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
