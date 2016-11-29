PATH = '../data/'
RAW_FILE = 'diabetic_data_initial.csv'
TEST_RAW_FILE = 'test_diabetic_data_initial.csv'

import csv
import numpy as np

from feature_type import *

def load_raw_data(path_to_data_file, unfold=False):
    result = []

    data_reader = csv.reader(open(path_to_data_file, 'r'))
    feature_labels = data_reader.next()

    # now make a mapping of feature to index
    feature_to_index = {}
    for i, label in enumerate(feature_labels[2:]): # remove patient and encounter_id
        feature_to_index[label] = i

    for row in data_reader:
        result.append(map(str.strip, row))

    features_and_labels = np.array(result)

    features = features_and_labels[:, 2:-1] # same
    labels = features_and_labels[:, -1]

    if len(features) != len(labels):
        raise Exception('Error: len(features) not equal len(labels) in data')

    # unfolds all the categorical features, as defined in feature_type.py
    if unfold:
        features, feature_to_index = _unfold_categorical_features(features, feature_to_index)

    return features, labels

def _unfold_categorical_features(features, feature_to_index):
    print 'Starting to unfold data set. Initial num features is [{}].'.format(len(features[0]))

    # for each feature, modify the data object
    for e, feature in enumerate(categorical_features):
        index = feature_to_index[feature]
        feature_vec = features[:, index]

        unique_values = np.unique(feature_vec)
        print 'Processing Feature [{}] with [{}] unique values including [{}].'.format(feature, len(unique_values), unique_values[:5])

        # now create a new feature matrix for this feature
        new_feature_representation = np.zeros((len(feature_vec), len(unique_values)))

        for i, value in enumerate(feature_vec):
            new_feature_representation[i, np.where(unique_values == value)[0][0]] = 1
        new_feature_representation = np.array(new_feature_representation)

        # update each feature index
        for feature_two in feature_to_index:
            index_two = feature_to_index[feature_two]
            num_net_new_features = len(unique_values) - 1 # subtract one because we remove the original feature
            if index_two > index:
                index_two += num_net_new_features
                feature_to_index[feature_two] = index_two
            # finally, remove the current feature index from the list
        del feature_to_index[feature]

        # finally, update features!
        temp_features = np.append(features[:,:index], new_feature_representation, axis=1)
        features = np.append(temp_features, features[:,index+1:], axis=1)

        print 'New total num features is [{}] after iteration [{}].'.format(len(features[0]), e)

    return features, feature_to_index

features, labels = load_raw_data(PATH + RAW_FILE, unfold=True)
