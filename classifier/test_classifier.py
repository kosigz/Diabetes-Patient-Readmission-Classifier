import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import StratifiedKFold

from preprocessing.preprocess import get_preprocessed_data as get_data, balance_samples

NROWS = 2000
# import some test data
ds = datasets.load_iris()
X, Y = ds.data, ds.target

X, Y = get_data("diabetic_data_initial.csv", nrows=NROWS)
# delete one of the columns which causes trouble with small data sets
if NROWS < 20000:
    X = X.drop('payer_code', 1)

# I guess we can keep this on hand to test trees or other classifiers that need categorical variables
categorical_X, categorical_Y = get_data("diabetic_data_initial.csv", nrows=NROWS, unfold=False)

#X, Y = balance_samples(X, Y)
#X, Y = pd.DataFrame(X), pd.DataFrame(Y)

def test_classifier_accuracy(classifier, folds=1):
    acc = []

    for i in range(folds):
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=1)
        classifier.train(train_X, train_Y)
        acc.append(classifier.accuracy(test_X, test_Y))

    return np.mean(acc)

def test_classifier(classifier, folds=1):
    print "{classifier} achieved {accuracy:2.2f}% accuracy on generated test data".format(
        classifier=classifier,
        accuracy=100 * test_classifier_accuracy(classifier, folds))

def test_classifier_accuracy_with_num_records(classifier, num=None, folds=10, categorical=False):
    accuracies = []
    if not num:
        num = len(X)
    if categorical:
        local_X, local_Y = categorical_X[:num], categorical_Y[:num]
    else:
        local_X, local_Y = X[:num], Y[:num]
#    skf = StratifiedKFold(n_splits=folds)

    print local_X.shape
    print local_Y.shape

#    for train_indices, test_indices in skf.split(local_X, local_Y):
#        train_X, train_Y = local_X.iloc[train_indices], local_Y.iloc[train_indices]
#        test_X, test_Y = local_X.iloc[test_indices], local_Y.iloc[test_indices]
#        accuracies.append(_test_classifier_accuracy(classifier, train_X, train_Y, test_X, test_Y))

    return np.mean(accuracies)

# inject your own data
def _test_classifier_accuracy(classifier, train_X, train_Y, test_X, test_Y):
    classifier.train(train_X, train_Y)
    return classifier.accuracy(test_X, test_Y)
