from scipy.stats import mode
import numpy as np

from . import VoteClassifier, SVMClassifier, test_classifier



class BaggingClassifier(VoteClassifier):
    """Classifier which uses bagging to account for class distribution skew"""
    def __init__(self, bags, Classifier):
        super(BaggingClassifier, self).__init__()
        self._classifiers = [Classifier() for i in range(bags)]
        self.type = "Bagged (10 x {})".format(str(self._classifiers[0]))

    # train all sub-classifiers on a provided dataset
    def _train(self, X, Y):
        # combine features with outputs for simpler row manipulation
        data = np.hstack((X, Y.reshape((-1, 1))))

        # unique classes, class index of each point, class counts
        labels, label_counts = np.unique(Y, return_counts=True)
        num_each = max(label_counts)
        diffs = [num_each - count for count in label_counts]

        # train the classifier in each bag with equal-sized samples
        for classifier in self._classifiers:
            train_data = data

            # add diff samples of each class to train_data
            for label, diff in zip(labels, diffs):
                if diff:
                    subset = data[data[:,-1] == label]
                    sample_idxs = np.random.choice(subset.shape[0], size=diff)
#                    print subset
#                    print sample_idxs
#                    print subset[sample_idxs]
                    # if it's a minority class, take a random oversample
                    train_data = np.vstack((train_data, subset[sample_idxs]))

            classifier.train(train_data[:,:-1], train_data[:,-1])



test_classifier(BaggingClassifier(10, SVMClassifier))