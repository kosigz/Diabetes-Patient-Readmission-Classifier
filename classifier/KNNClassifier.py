from sklearn.neighbors import KNeighborsClassifier

from . import AbstractZnormClassifier



class KNNClassifier(AbstractZnormClassifier):
    """Classifier which uses the K-nearest neighbor algorithm"""
    def __init__(self, k, **kwargs):
        # keyword arguments are passed on to scikit-learn's KNN implementation
        # see http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        # relevant kwargs (* indicates default):
        #     weights (string): "uniform"*, "distance"
        #     p (int): 1 (Manhattan distance), 2* (Euclidean distance)
        #     n_jobs (int): 1* or more (cores used to parallelize neighbor search)
        super(KNNClassifier, self).__init__("KNN", k=k, **kwargs)
        self._knn = KNeighborsClassifier(n_neighbors=k, **kwargs)

    # train a KNN classifier on a provided dataset
    def _train(self, X, Y):
        self._knn.fit(X, Y)

    # classify a set of test points
    def _classify(self, test_X):
        return self._knn.predict(test_X)



#from . import test_classifier, SMOTEClassifier

#for k in (1, 3, 5, 10):
#    for w in ("uniform", "distance"):
#        for p in (1, 2):
#            test_classifier(KNNClassifier(k, weights=w, p=p, n_jobs=4), num_samples=5000)
#for k in (3, 5, 10, 15):
#    test_classifier(SMOTEClassifier(KNNClassifier(k, n_jobs=4)))