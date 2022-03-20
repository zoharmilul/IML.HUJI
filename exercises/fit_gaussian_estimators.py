from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    quiz_arr = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print(UnivariateGaussian.log_likelihood(1,1,quiz_arr))
    print(UnivariateGaussian.log_likelihood(10,1,quiz_arr))
    true_exp = 10
    true_var = 1
    samples = np.random.normal(true_exp, true_var, 1000)
    uni_gauss = UnivariateGaussian().fit(samples)
    print(uni_gauss.mu_, uni_gauss.var_)

    # raise NotImplementedError()

    # Question 2 - Empirically showing sample mean is consistent
    intervals = [i * 10 for i in range(1, 101)]
    diff = []
    myUniGauss = UnivariateGaussian()
    for i in range(100):
        temp = myUniGauss.fit(samples[:(i + 1) * 10])
        diff.append(np.abs(temp.mu_ - true_exp))

    go.Figure([go.Scatter(x=intervals, y=diff, mode='markers+lines', name=r'abs diff')],
              layout=go.Layout(title=r"$\text{Absolute distance between the estimated- and true value of the "
                                     r"expectation}$",
                               xaxis_title="$\\text{Number of samples}$",
                               yaxis_title="r$Distance$")).show()

    # raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdfArr = uni_gauss.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=pdfArr, mode='markers', name=r'abs diff')],
              layout=go.Layout(title=r"$\text{PDF graph of fitted Gaussian Estimator}$",
                               xaxis_title="$\\text{Sample Value}$",
                               yaxis_title="r$PDF$")).show()

    # raise NotImplementedError()


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(mu, cov, 1000)
    multi_var_gauss = MultivariateGaussian().fit(samples)
    print(multi_var_gauss.mu_)
    print(multi_var_gauss.cov_)
    # raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    max = np.NINF
    argmax = (0, 0)
    log_likley_arr = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            temp = np.transpose([f1[i], 0, f3[j], 0])
            log_likley_arr[i][j] = MultivariateGaussian().log_likelihood(temp, cov, samples)
            if log_likley_arr[i][j] > max:
                argmax = (i, j)
                max = log_likley_arr[i][j]

    go.Figure([go.Heatmap(x=f3, y=f1, z=log_likley_arr)],
              layout=go.Layout(title=r"$\text{Log Likelihood for Different Modules}$",
                               xaxis_title="$\\text{Function 1 evenly distributed}$",
                               yaxis_title="r$\\text{Function 3 evenly distributed}$")).show()
    # Question 6 - Maximum likelihood
    print("Maximum value achieved at: [{f1},0,{f3},0]".format(f1=f1[argmax[0]], f3=f3[argmax[1]]))
    print("Maximum value is: {fmax}".format(fmax = np.round(max, 3)))
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
