from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, amount_in_clas = np.unique(y, return_counts=True)
        num_class = self.classes_.shape[0]
        num_features = X.shape[1]
        num_samples = y.shape[0]
        self.mu_ = np.ndarray(shape=(num_class, num_features))
        self.vars_ = np.ndarray(shape=(num_class, num_features))
        self.pi_ = np.ndarray(num_samples)
        for i, k in enumerate(self.classes_):
            xk = X[np.where(y == k)]
            self.mu_[i] = np.mean(xk, axis=0)
            self.pi_[i] = xk.shape[0] / y.shape[0]
            self.vars_[i] = np.var(xk, axis=0, ddof=1)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = np.zeros(len(X))
        likely = self.likelihood(X)
        for i, samp in enumerate(likely):
            y_pred[i] = np.argmax(samp)
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likely = []
        for k in range(len(self.classes_)):
            scalar = np.log(self.pi_[k])
            temp_sig = []
            for i in range(X.shape[1]):
                temp_sig.append(np.log(np.sqrt(2 * np.pi * self.vars_[k][i])) +
                                ((np.square(X[:, i] - self.mu_[k][i])) / self.vars_[k][i]) / 2)
            likely.append(scalar - np.sum(np.array(temp_sig), axis=0))

        return np.array(likely).T

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
