from sklearn.neighbors import KNeighborsClassifier

from . import AbstractZnormClassifier, test_classifier



class KNNClassifier(AbstractZnormClassifier):
    """Classifier which uses the K-nearest neighbor algorithms"""
    def __init__(self, k):
        super(AbstractZnormClassifier, self).__init__("KNN", k=k)
        self.knn = KNeighborsClassifier(n_neighbors=self.params["k"])

    # train a KNN classifier on a provided dataset
    def _train(self, X, Y):
        self.knn.fit(X, Y)

    # classify a set of test points
    def _classify(self, test_X):
        return self.knn.predict(test_X)

test_classifier(KNNClassifier(10))