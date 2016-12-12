from sklearn.ensemble import RandomForestClassifier as RandomForest

from . import AbstractZnormClassifier, test_classifier



class RandomForestClassifier(AbstractZnormClassifier):
    """Classifier which uses the random forests"""
    def __init__(self, n=10, **kwargs):
        # keyword arguments are passed on to scikit-learn's KNN implementation
        # see http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        # relevant kwargs (* indicates default):
        #     n (int): 10* (number of trees in the random forest)
        #     n_jobs (int): 1* or more (cores used to parallelize neighbor search)
        super(RandomForestClassifier, self).__init__("Random Forest", n=n, **kwargs)
        self._forest = RandomForest(n_estimators=n, **kwargs)

    # build a random forest for a provided dataset
    def _train(self, X, Y):
        self._forest.fit(X, Y)

    # classify a set of test points
    def _classify(self, test_X):
        return self._forest.predict(test_X)



#for n in (5, 10, 25):
#    test_classifier(RandomForestClassifier(n))