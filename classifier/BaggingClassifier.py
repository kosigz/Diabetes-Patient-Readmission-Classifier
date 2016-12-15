from scipy.stats import mode
import numpy as np

from . import VoteClassifier, SMOTEClassifier



class BaggingClassifier(VoteClassifier):
    """Classifier which uses bagging to account for class distribution skew"""
    def __init__(self, bags, Classifier, BalancedClassifier=SMOTEClassifier):
        super(BaggingClassifier, self).__init__(
            *(BalancedClassifier(Classifier()) for i in range(bags)))
        self.type = "Bagged ({} x {})".format(bags, str(self._classifiers[0]))



#from . import SVMClassifier, test_classifier
'''
test_classifier(SVMClassifier(5), folds=5)
test_classifier(
    BaggingClassifier(3, lambda: SVMClassifier(5), SMOTEClassifier),
    folds=5)
test_classifier(
    BaggingClassifier(5, lambda: SMOTEClassifier(SVMClassifier(C=6, kernel="rbf"))),
    folds=5)
'''
test_classifier(BaggingClassifier(5, lambda: SMOTEClassifier(RandomForestClassifier(128))), folds=5)

'''
test_classifier(
    VoteClassifier(
        BaggingClassifier(5, lambda: SMOTEClassifier(SVMClassifier(C=9, kernel="poly", degree=3))),
        BaggingClassifier(5, lambda: SMOTEClassifier(SVMClassifier(C=6, kernel="rbf"))),
        BaggingClassifier(5, lambda: SMOTEClassifier(RandomForestClassifier(128)),
        BaggingClassifier(10, lambda: SMOTEClassifier(RandomForestClassifier(128)),
    ))),
    folds=5)
'''
