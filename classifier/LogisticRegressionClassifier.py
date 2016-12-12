from sklearn.linear_model import LogisticRegression

from . import AbstractZnormClassifier, test_classifier



class LogisticRegressionClassifier(AbstractZnormClassifier):
    """Classifier which uses one-vs-one support vector machines"""
    def __init__(self, C=1, **kwargs):
        # keyword arguments are passed on to scikit-learn's SVM implementation
        # see http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        # relevant kwargs (* indicates default):
        #     penalty (string): "l1" or "l2"* (norm to regularize against)
        #     C (float): 1* (inverse of regularization strength)
        #     n_jobs (int): 1* or more (cores used to parallelize CV; -1 for all)
        super(LogisticRegressionClassifier, self).__init__(
            "Logistic Regression", C=C, **kwargs)
        self._lr = LogisticRegression(C=self.params["C"], **kwargs)

    # train a KNN classifier on a provided dataset
    def _train(self, X, Y):
        self._lr.fit(X, Y)

    # classify a set of test points
    def _classify(self, test_X):
        return self._lr.predict(test_X)

for C in (0.01, 0.1, 1, 2, 5, 10, 25):
    for penalty in ("l1", "l2"):
        test_classifier(LogisticRegressionClassifier(C, penalty=penalty, n_jobs=4))