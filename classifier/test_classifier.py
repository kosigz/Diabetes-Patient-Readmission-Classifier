import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import StratifiedKFold

from preprocessing.preprocess import get_preprocessed_data as get_data, balance_samples

# NROWS = 2000
# import some test data
ds = datasets.load_iris()
X, Y = ds.data, ds.target

X, Y = get_data("diabetic_data_initial.csv")
# delete one of the columns which causes trouble with small data sets

# I guess we can keep this on hand to test trees or other classifiers that need categorical variables
# categorical_X, categorical_Y = get_data("diabetic_data_initial.csv", nrows=NROWS, unfold=False)

#X, Y = balance_samples(X, Y)
#X, Y = pd.DataFrame(X), pd.DataFrame(Y)

def test_classifier_accuracy(classifier, folds=10, num_samples=None):
    acc = []

    # can use to create training curves, etc.
    if not num_samples:
        num_samples = len(X)
    temp_X = X[:num_samples]
    temp_Y = Y[:num_samples]

    for i in range(folds):
        train_X, test_X, train_Y, test_Y = train_test_split(temp_X, temp_Y, random_state=42)
        classifier.train(train_X, train_Y)
        acc.append(classifier.accuracy(test_X, test_Y))

    return np.mean(acc)

def test_classifier(classifier, folds=1, num_samples=2000):
    print "{classifier} achieved {accuracy:2.2f}% accuracy on generated test data".format(
        classifier=classifier,
        accuracy=100 * test_classifier_accuracy(classifier, folds, num_samples))

# inject your own data
def _test_classifier_accuracy(classifier, train_X, train_Y, test_X, test_Y):
    classifier.train(train_X, train_Y)
    return classifier.accuracy(test_X, test_Y)
