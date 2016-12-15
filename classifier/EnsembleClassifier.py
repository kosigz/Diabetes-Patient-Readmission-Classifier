from scipy.stats import mode
import numpy as np

from . import (AbstractClassifier, BaggingClassifier, VoteClassifier,
    KNNClassifier, SVMClassifier, RandomForestClassifier,
    LogisticRegressionClassifier, test_classifier)



def EnsembleClassifier(bags, *Classifiers):
    return VoteClassifier(*[BaggingClassifier(bags, C) for C in Classifiers])


'''
classifier_fns = (
    lambda: SMOTEClassifier(KNNClassifier(1)),
    lambda: SMOTEClassifier(KNNClassifier(2)),
    lambda: SMOTEClassifier(SVMClassifier(6, kernel='rbf')),
    lambda: SMOTEClassifier(SVMClassifier(8, kernel='poly', degree=3)),
    lambda: SMOTEClassifier(RandomForestClassifier(k=128))
    )

bagged_classifiers = [BaggingClassifier(10, C) for C in classifier_fns]
ensemble = VoteClassifier(*bagged_classifiers)
test_classifier(VoteClassifier(*bagged_classifiers))
'''
#test_classifier(SMOTEClassifier(SVMClassifier(6, kernel='rbf')))
