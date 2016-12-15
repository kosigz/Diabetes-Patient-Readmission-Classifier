from sklearn.ensemble import RandomForestClassifier as RandomForest

from . import AbstractZnormClassifier



class RandomForestClassifier(AbstractZnormClassifier):
    """Classifier which uses the random forests"""
    def __init__(self, n=10, **kwargs):
        # keyword arguments are passed on to scikit-learn's KNN implementation
        # see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
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



#from . import test_classifier, BaggingClassifier, OversampleClassifier
##
#for bags in (3, 5, 10):
#    for n in (5, 10, 25):
#        test_classifier(
#            BaggingClassifier(bags, lambda: RandomForestClassifier(n, n_jobs=4)),
#            unfold=false)
#test_classifier(
#    BaggingClassifier(10,
#        lambda: RandomForestClassifier(128, class_weight="balanced"),
#        BalancedClassifier=(lambda x: x)),
#    folds=10,
#    num_samples=5000,
#    unfold=False)