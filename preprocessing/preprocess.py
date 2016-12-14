from os import path
import pandas as pd
import numpy as np
import re
from imblearn.combine import SMOTEENN

from feature_types import feature_types as ftypes

DATA_PATH = "../data/"
DATA_PATH = "data/"
OUTPUT_PATH = "../output/"
DATA_FILE = "diabetic_data_initial.csv"

def main():
    features, labels = get_preprocessed_data(DATA_FILE, nrows=10000)
    print features.columns

def get_preprocessed_data(csv_path, nrows=None, unfold=True):
    data = load_data(csv_path, nrows=nrows)
    X, y = preprocess(data.iloc[:,:-1], unfold=unfold), preprocess_labels(data.iloc[:,-1])
    return X, y

# read CSV data file into a Pandas DataFrame
def load_data(csv_path, nrows=None):
    return pd.read_csv(
        path.join(DATA_PATH, csv_path),
        nrows=nrows,
        na_values=("?", "None", "Other"),
        converters=dict([(name, convert_range) for name in ftypes["range"]] +
            [("diag_%d" % i, convert_diag) for i in (1, 2, 3)]))

# convert a range such as "[10-20)" into its numeric midpoint
def convert_range(range_str):
    matches = re.match(r"\[(\d+)-(\d+)\)", range_str)
    if matches:
        return np.mean(map(int, matches.groups()))

    return np.nan

# convert a diagnosis code into its group name
# Reference: https://www.hindawi.com/journals/bmri/2014/781670/tab2/
def convert_diag(diag_str):
    try:
        diag = float(diag_str)

        if np.floor(diag) == 250:
            return "Diabetes"
        elif 390 <= diag <= 459 or diag == 785:
            return "Circulatory"
        elif 460 <= diag <= 519 or diag == 786:
            return "Respiratory"
        elif 520 <= diag <= 579 or diag == 787:
            return "Digestive"
        elif 580 <= diag <= 629 or diag == 788:
            return "Genitourinary"
        elif 710 <= diag <= 739:
            return "Musculoskeletal"
        elif 800 <= diag <= 999:
            return "Injury"
        elif (1 <= diag <= 279 or
              290 <= diag <= 319 or
              680 <= diag <= 709 or
              780 <= diag <= 782 or
              790 <= diag <= 799 or
              diag == 784):
            return "Neoplasms"
        else:
            return "Other"
    except:
        pass

    return np.nan

# fill missing continuous values with the feature average
def impute_mean(fts):
    return fts.fillna(fts.mean())

# converts all features to continuous values and imputes missing values for
# continuous features
def preprocess(fts, unfold=True):
    continuous = impute_mean(fts[ftypes["continuous"] + ftypes["range"]])
    categorical = fts[ftypes["categorical"]]
    if unfold:
        categorical = pd.get_dummies(categorical, prefix_sep=" = ")

    return pd.concat([continuous, categorical], axis=1)

def preprocess_labels(labels):
    labels = labels.replace(['<30'], 1).replace(['NO', '>30'], 0)
    # the instructions state that the variance of the following lines should be
    # printed by the program as a sanity check
#    print 'total number of instances by class (0 = >30, NO; 1 = <30)\n{}'.format(labels.value_counts())
    return labels

def balance_samples(X, y):
    '''
    This turned out to be a pain. Will use library.
    negative_class_indices = y == 0
    positive_class_indices = y == 1
    print negative_class_indices
    print positive_class_indices
    diff = negative_class_indices.sum() - positive_class_indices.sum()
    if diff <= 0:
        return X, y

    positive_indices = []
    # this is a hack
    for i, b in enumerate(positive_class_indices):
        if b:
            positive_indices.append(i)

    indices_to_sample = np.random.choice(positive_indices, size=diff)
    Xprime, yprime = X.append(X.iloc[indices_to_sample]), y.append(y.iloc[indices_to_sample])
    # now resample
    return Xprime, yprime
    '''
    sm = SMOTEENN()
    return sm.fit_sample(X, y)
