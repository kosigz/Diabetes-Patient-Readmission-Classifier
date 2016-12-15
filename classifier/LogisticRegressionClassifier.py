from sklearn.linear_model import LogisticRegression
import numpy as np

from . import AbstractZnormClassifier



class LogisticRegressionClassifier(AbstractZnormClassifier):
    """Classifier which uses regularized logistic regression"""
    def __init__(self, C=1, phi=None, degree=3, **kwargs):
        # keyword arguments are passed on to scikit-learn's SVM implementation
        # see http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        # relevant kwargs (* indicates default):
        #     C (float): 1* (inverse of regularization strength)
        #     penalty (string): "l1" or "l2"* (norm to regularize against)
        #     n_jobs (int): 1* or more (cores used to parallelize CV; -1 for all)
        super(LogisticRegressionClassifier, self).__init__(
            "Logistic Regression", C=C, phi=phi, degree=degree, **kwargs)
        self._lr = LogisticRegression(C=C, **kwargs)

        if phi is None or (phi == "poly" and degree == 1):
            self.phi = lambda X: X
        elif phi == "poly":
            self.phi = lambda X: poly_expand(X, degree)
        else:
            self.phi = phi

    # train a logistic regression model on a provided dataset
    def _train(self, X, Y):
        self._lr.fit(self.phi(X), Y)

    # classify a set of test points
    def _classify(self, test_X):
        return self._lr.predict(self.phi(test_X))



# perform Nth-degree polynomial feature basis expansion
def poly_expand(X, n):
    ft_powers = np.array([X ** i for i in np.arange(n) + 1])
    return np.swapaxes(ft_powers, 0, 1).reshape((X.shape[0], -1))



#from . import test_classifier
#
#for C in (0.01, 0.1, 1, 2, 5, 10, 25):
#    for penalty in ("l1", "l2"):
#        test_classifier(LogisticRegressionClassifier(
#            C=C, penalty=penalty, class_weight="balanced"))