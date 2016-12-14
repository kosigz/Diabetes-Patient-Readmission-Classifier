from scipy.stats import mode
import numpy as np

from . import VoteClassifier, BalancedClassifier, KNNClassifier, test_classifier



class BaggingClassifier(VoteClassifier):
    """Classifier which uses bagging to account for class distribution skew"""
    def __init__(self, bags, Classifier):
        super(BaggingClassifier, self).__init__(
            *(BalancedClassifier(Classifier()) for i in range(bags)))
        self.type = "Bagged (10 x {})".format(str(self._classifiers[0]))



test_classifier(BaggingClassifier(3, lambda: KNNClassifier(5)), folds=5)