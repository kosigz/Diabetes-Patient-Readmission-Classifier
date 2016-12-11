from . import AbstractClassifier, znorm_dec as znorm



class AbstractZnormClassifier(AbstractClassifier):
    """Arbitrary classifier implementation with automatic z-score normalization"""

    # perform any necessary normalization, store data, and train the model
    @znorm
    def train(self, X, Y, normalize):
        self.X, self.Y, self.normalize = X, Y, normalize
        return self._train(X, Y)

    def classify(self, test_X):
        return self._classify(self.normalize(test_X))