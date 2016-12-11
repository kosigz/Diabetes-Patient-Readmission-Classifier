from abc import abstractmethod



class AbstractClassifier(object):
    """Arbitrary classifier implementation with helper methods"""
    def __init__(self, typ, classes, **params):
        self.type = type       # classifier type ex: "knn", "svm", etc
        self.params = params   # hyperparameters ex: { "lambda": 1 }
        self.classes = classes # output classes ex: [0, 1, 2]


    # actually train the model and store any necessary data for the classifier
    @abstractmethod
    def _train(X, Y):
        pass

    # perform any necessary normalization, store data, and train the model
    def train(self, X, Y):
        self.X, self.Y = X, Y
        return self._train(X, Y)

    # classify a set of test points; must accept an ndarray with shape (M, D)
    # where M is the number of test samples and D is the number of features; if
    # the classifier can only act on one point at a time, it must loop through
    # each sample
    @abstractmethod
    def classify(self, test_X):
        pass

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
        return self.correct(test_X, test_Y) / test_X.shape[0]