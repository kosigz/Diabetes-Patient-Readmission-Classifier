from scipy.stats import mode
import numpy as np

from . import (AbstractClassifier, BaggingClassifier, VoteClassifier,
    KNNClassifier, SVMClassifier, RandomForestClassifier,
    LogisticRegressionClassifier, test_classifier)



def EnsembleClassifier(bags, *Classifiers):
    return VoteClassifier(*[BaggingClassifier(bags, C) for C in Classifiers])



classifier_fns = (
    lambda: KNNClassifier(3),
    lambda: KNNClassifier(10),
    SVMClassifier,
    RandomForestClassifier,
    LogisticRegressionClassifier)

bagged_classifiers = [BaggingClassifier(10, C) for C in classifier_fns]
ensemble = VoteClassifier(*bagged_classifiers)
test_classifier(VoteClassifier(*bagged_classifiers))