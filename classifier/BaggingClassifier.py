from scipy.stats import mode
import numpy as np

from . import VoteClassifier, SMOTEClassifier, SVMClassifier, test_classifier



class BaggingClassifier(VoteClassifier):
    """Classifier which uses bagging to account for class distribution skew"""
    def __init__(self, bags, Classifier, BalancedClassifier=SMOTEClassifier):
        super(BaggingClassifier, self).__init__(
            *(BalancedClassifier(Classifier()) for i in range(bags)))
        self.type = "Bagged ({} x {})".format(bags, str(self._classifiers[0]))


test_classifier(SVMClassifier(5), folds=5)
test_classifier(
    BaggingClassifier(3, lambda: SVMClassifier(5), SMOTEClassifier),
    folds=5)