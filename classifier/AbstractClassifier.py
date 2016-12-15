from abc import abstractmethod
from imblearn.combine import SMOTEENN

class AbstractClassifier(object):
    """Arbitrary classifier implementation with helper methods"""
    def __init__(self, typ, **params):
        self.type = typ        # classifier type ex: "knn", "svm", etc
        self.params = params   # hyperparameters ex: { "lambda": 1 }
        self.trained = False   # record when the model has been trained

    def __str__(self):
        return "{ctype} classifier with [{params}]".format(
            ctype=self.type,
            params=", ".join("{} = {}".format(*p) for p in self.params.iteritems()))


    # actually train the model and store any necessary data for the classifier
    @abstractmethod
    def _train(self, X, Y):
        pass

    # perform any necessary normalization, store data, and train the model
    def train(self, X, Y):
        self.X, self.Y, self.trained = X, Y, True
#        print str(self)
        return self._train(X, Y)


    # classify a set of test points; must accept an ndarray with shape (M, D)
    # where M is the number of test samples and D is the number of features; if
    # the classifier can only act on one point at a time, it must loop through
    # each sample
    @abstractmethod
    def _classify(self, test_X):
        pass

    # perform any necessary normalization and test the model
    def classify(self, test_X):
        if not self.trained:
            raise RuntimeError(
                "Unable to perform classification; model must first be trained")

        return self._classify(test_X)


    # list of output classes ex: [0, 1, 2]
    @property
    def classes(self):
        return self.Y.unique()

    # list of binary {-1, 1} class labels for OVA on each output class
    @property
    def binary_classes(self):
        return [(self.Y == c).astype(int) * 2 - 1 for c in self.classes]


    # number of output classes
    @property
    def L(self):
        return len(self.classes)

    # number of training samples
    @property
    def N(self):
        return self.X.shape[0]

    # number of features
    @property
    def D(self):
        return self.X.shape[1]


    # counts the number of correctly classified test points
    def correct(self, test_X, test_Y):
        return (self.classify(test_X) == test_Y).sum()

    # calculates the proportion of correctly classified test points (0-1)
    def accuracy(self, test_X, test_Y):
        return self.correct(test_X, test_Y) / float(test_X.shape[0])
