import numpy as np
from scipy.stats import mode

from . import AbstractClassifier



class ModeClassifier(AbstractClassifier):
    """Classifier which naively guesses the most common class for all test samples"""
    def __init__(self):
        super(ModeClassifier, self).__init__("Mode")

    # train a KNN classifier on a provided dataset
    def _train(self, X, Y):
        self._mode = mode(Y).mode[0]

    # classify a set of test points
    def _classify(self, test_X):
        return np.full(test_X.shape[0], self._mode)



#from . import SMOTEClassifier, test_classifier
#for k in (3, 5, 10, 20):
#    for w in ("uniform", "distance"):
#        for p in (1, 2):
#            test_classifier(ModeClassifier(k, weights=w, p=p, n_jobs=4))
# -*- coding: utf-8 -*-

#test_classifier(ModeClassifier())