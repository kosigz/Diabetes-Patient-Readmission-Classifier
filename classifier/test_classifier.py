import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold

from preprocessing.preprocess import get_preprocessed_data as get_data

NROWS = 10000
# import some test data
# ds = datasets.load_iris()
# X, Y = ds.data, ds.target
X, Y = get_data('diabetic_data_initial.csv', nrows=NROWS)

# delete one of the columns which causes trouble with small data sets
if NROWS < 20000:
    X = X.drop('payer_code', 1)

#X, Y = datasets.make_blobs(n_samples=1000, centers=[[0,0],[2,0],[0,2]],
#                           n_features=2, cluster_std=1, random_state=1)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=1)

print 'Null count: {}'.format(train_X.isnull().sum().sum())

def test_classifier_accuracy(classifier):
    classifier.train(train_X, train_Y)
    return classifier.accuracy(test_X, test_Y)

def test_classifier(classifier):
    print "{classifier} achieved {accuracy:2.2f}% accuracy on generated test data".format(
        classifier=classifier,
        accuracy=100 * test_classifier_accuracy(classifier))

def test_classifier_accuracy_with_num_records(classifier, num, folds=10):
    if num > NROWS:
        raise Exception('Not possible.')

    accuracies = []
    local_X, local_Y = X[:num], Y[:num]
    skf = StratifiedKFold(n_splits=folds)

    for train_indices, test_indices in skf.split(local_X, local_Y):
        train_X, train_Y = local_X.iloc[train_indices], local_Y.iloc[train_indices]
        test_X, test_Y = local_X.iloc[test_indices], local_Y.iloc[test_indices]
        accuracies.append(_test_classifier_accuracy(classifier, train_X, train_Y, test_X, test_Y))

    return np.mean(accuracies)

# inject your own data
def _test_classifier_accuracy(classifier, train_X, train_Y, test_X, test_Y):
    classifier.train(train_X, train_Y)
    return classifier.accuracy(test_X, test_Y)
