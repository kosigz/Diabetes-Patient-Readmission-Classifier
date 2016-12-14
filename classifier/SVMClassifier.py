from sklearn.svm import SVC

from . import AbstractZnormClassifier, test_classifier

class SVMClassifier(AbstractZnormClassifier):
    """Classifier which uses one-vs-one support vector machines"""
    def __init__(self, C=1, **kwargs):
        # keyword arguments are passed on to scikit-learn's SVM implementation
        # see http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        # relevant kwargs (* indicates default):
        #     C (float): 1* (penalty term)
        #     kernel (string): "linear", "poly", "rbf"*, "sigmoid"
        #     degree (int): 1, 2, 3*, ... (degree for polynomial kernel)
        super(SVMClassifier, self).__init__("SVM", C=C, **kwargs)
        self._svm = SVC(C=C, **kwargs)

    # train a SVM classifier on a provided dataset
    def _train(self, X, Y):
        self._svm.fit(X, Y)

    # classify a set of test points
    def _classify(self, test_X):
        return self._svm.predict(test_X)

'''
for C in (0.01, 0.1, 1, 2, 5, 10, 25):
    for kernel in ("linear", "poly", "rbf", "sigmoid"):
        if kernel == "poly":
            for degree in range(2, 4):
                test_classifier(SVMClassifier(C, kernel=kernel, degree=degree, verbose=True))
        else:
            test_classifier(SVMClassifier(C, kernel=kernel, verbose=True))
'''
