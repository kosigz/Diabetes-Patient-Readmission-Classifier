from sklearn.neighbors import KNeighborsClassifier

from . import AbstractZnormClassifier



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

def gen_clusters(*clusters):
    data, outputs = np.empty(0), np.empty(0)
    for i, (mean, stdev, n) in enumerate(clusters):
        data = np.append(data, np.random.normal(mean, stdev, n))
        outputs = np.append(outputs, [i] * n)
    return data[:,None], outputs

import numpy as np
X, Y = gen_clusters((0, 0.5, 100),
                    (1, 0.25, 100),
                    (2, 1, 150))
test_X, test_Y = gen_clusters((0, 0.25, 20),
                    (1, 0.125, 10),
                    (2, 0.5, 30))
classifier = KNNClassifier(3)
classifier.train(X, Y)
print classifier
print classifier.accuracy(test_X, test_Y)