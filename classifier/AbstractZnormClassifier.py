from . import AbstractClassifier
import znorm



class AbstractZnormClassifier(AbstractClassifier):
    """Arbitrary classifier implementation with automatic z-score normalization"""
    # perform any necessary normalization, store data, and train the model
    @znorm.znorm_dec
    def train(self, X, Y, normalize):
        self.normalize = normalize
        return super(AbstractZnormClassifier, self).train(X, Y)

    # normalize the test data according to the training distribution and
    # perform classification
    def classify(self, test_X):
        if not self.trained:
            raise RuntimeError(
                "Unable to perform classification; model must first be trained")
        return super(AbstractZnormClassifier, self).classify(self.normalize(test_X))