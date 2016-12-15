from scipy.stats import mode
import numpy as np
from imblearn.combine import SMOTEENN

from . import AbstractClassifier



class OversampleClassifier(AbstractClassifier):
    """Classifier which uses minority oversampling to balance minority class"""
    def __init__(self, classifier, **kwargs):
        super(OversampleClassifier, self).__init__(
            "Oversampled ({})".format(classifier), **kwargs)
        self.classifier = classifier

    # balance the dataset, then train on it
    def _train(self, X, Y):
        # combine features with outputs for simpler row manipulation
        data = np.hstack((X, Y.reshape((-1, 1))))

        # unique classes, class index of each point, class counts
        labels, label_counts = np.unique(Y, return_counts=True)
        num_each = max(label_counts)
        diffs = [num_each - count for count in label_counts]

        # add diff samples of each class to train_data
        for label, diff in zip(labels, diffs):
            if diff:
                subset = data[data[:,-1] == label]
                sample_idxs = np.random.choice(subset.shape[0], size=diff)
#                    print subset
#                    print sample_idxs
#                    print subset[sample_idxs]
                # if it's a minority class, take a random oversample
                data = np.vstack((data, subset[sample_idxs]))


        # train the classifier on equally-distributed samples
        self.classifier.train(data[:,:-1], data[:,-1])

    # classify a set of test points
    def _classify(self, test_X):
        return self.classifier.classify(test_X)

sm = SMOTEENN()
class SMOTEClassifier(OversampleClassifier):
    """Classifier which uses bagging to account for class distribution skew"""
    def __init__(self, classifier, **kwargs):
        super(OversampleClassifier, self).__init__(
            "SMOTE ({})".format(classifier), **kwargs)
        self.classifier = classifier

    # balance the dataset, then train on it
    def _train(self, X, Y):
        # train the classifier on SMOTE-balanced samples
        self.classifier.train(*sm.fit_sample(X, Y))


#from . import SVMClassifier, RandomForestClassifier, LogisticRegressionClassifier, KNNClassifier, test_classifier
'''
for kernel in ['rbf', 'poly']:
    for c in range(1, 10, 1):
        if kernel == poly:
            test_classifier(SMOTEClassifier(SVMClassifier(C=c, kernel=kernel, degree=3)), folds=10)
        test_classifier(SMOTEClassifier(SVMClassifier(C=c, kernel=kernel, degree=3)), folds=10)
'''
'''
for t in [4, 8, 16, 25, 32, 50, 64, 75, 100, 128]:
    test_classifier(SMOTEClassifier(RandomForestClassifier(t)))
'''
'''
for c in [0.1, 0.5, 1, 1.5, 2.5]:
    for degree in range(2, 4):
        test_classifier(SMOTEClassifier(LogisticRegressionClassifier(C=c, degree=degree)))
raise Exception('done for now')
'''
'''
for k in [1, 2, 4, 8, 16, 25, 32, 64]:
    test_classifier(SMOTEClassifier(KNNClassifier(k)))
raise Exception('done for now')
'''
#for t in [4, 8, 16, 25, 32, 50, 64, 75, 100, 128]:
#    test_classifier(RandomForestClassifier(t))

#test_classifier(SMOTEClassifier(SVMClassifier(C=6, kernel='rbf')), folds=10)
