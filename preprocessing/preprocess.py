from os import path
import pandas as pd
import numpy as np
import re

from feature_types import feature_types as ftypes

DATA_PATH = "../data/"
OUTPUT_PATH = "../output/"
DATA_FILE = "diabetic_data_initial.csv"

def main():
    features, labels = get_preprocessed_data(DATA_FILE, nrows=1000)
    print features.columns

def get_preprocessed_data(csv_path, nrows=None):
    data = load_data(csv_path, nrows=nrows)
    return preprocess(data.iloc[:,:-1]), data.iloc[:,-1]

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


PATH = '../data/'
RAW_FILE = 'diabetic_data_initial.csv'
TEST_RAW_FILE = 'test_diabetic_data_initial.csv'

import csv

#from feature_type import *

#def load_raw_data(path_to_data_file, unfold=False):
#    result = []
#
#    data_reader = csv.reader(open(path_to_data_file, 'r'))
#    feature_labels = data_reader.next()
#
#    # now make a mapping of feature to index
#    feature_to_index = {}
#    for i, label in enumerate(feature_labels[2:]): # remove patient and encounter_id
#        feature_to_index[label] = i
#
#    for row in data_reader:
#        result.append(map(str.strip, row))
#
#    features_and_labels = np.array(result)
#
#    features = features_and_labels[:, 2:-1] # same
#    labels = features_and_labels[:, -1]
#
#    if len(features) != len(labels):
#        raise Exception('Error: len(features) not equal len(labels) in data')
#
#    # unfolds all the categorical features, as defined in feature_type.py
#    if unfold:
#        features, feature_to_index = _unfold_categorical_features(features, feature_to_index)
#
#    return features, labels
#
#def _unfold_categorical_features(features, feature_to_index):
#    print 'Starting to unfold data set. Initial num features is [{}].'.format(len(features[0]))
#
#    # for each feature, modify the data object
#    for e, feature in enumerate(categorical_features):
#        index = feature_to_index[feature]
#        feature_vec = features[:, index]
#
#        unique_values = np.unique(feature_vec)
#        print 'Processing Feature [{}] with [{}] unique values including [{}].'.format(feature, len(unique_values), unique_values[:5])
#
#        # now create a new feature matrix for this feature
#        new_feature_representation = np.zeros((len(feature_vec), len(unique_values)))
#
#        for i, value in enumerate(feature_vec):
#            new_feature_representation[i, np.where(unique_values == value)[0][0]] = 1
#        new_feature_representation = np.array(new_feature_representation)
#
#        # update each feature index
#        for feature_two in feature_to_index:
#            index_two = feature_to_index[feature_two]
#            num_net_new_features = len(unique_values) - 1 # subtract one because we remove the original feature
#            if index_two > index:
#                index_two += num_net_new_features
#                feature_to_index[feature_two] = index_two
#            # finally, remove the current feature index from the list
#        del feature_to_index[feature]
#
#        # finally, update features!
#        temp_features = np.append(features[:,:index], new_feature_representation, axis=1)
#        features = np.append(temp_features, features[:,index+1:], axis=1)
#
#        print 'New total num features is [{}] after iteration [{}].'.format(len(features[0]), e)
#
#    return features, feature_to_index
#
##features, labels = load_raw_data(PATH + RAW_FILE, unfold=True)

if __name__ == '__main__':
    main()
