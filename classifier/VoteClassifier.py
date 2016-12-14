from scipy.stats import mode

from . import AbstractClassifier, KNNClassifier, SVMClassifier, RandomForestClassifier, test_classifier



class VoteClassifier(AbstractClassifier):
    """Ensemble classifier which uses a vote of sub-classifiers"""
    def __init__(self, *classifiers):
        super(VoteClassifier, self).__init__("Vote")
        self._classifiers = classifiers


    # train all sub-classifiers on a provided dataset
    def _train(self, X, Y):
        for c in self._classifiers:
            c.train(X, Y)

    # classify a set of test points
    def _classify(self, test_X):
        return mode(c.classify(test_X) for c in self._classifiers)



#test_classifier(VoteClassifier(
#    KNNClassifier(3),
#    KNNClassifier(10),
#    SVMClassifier(),
#    RandomForestClassifier()))