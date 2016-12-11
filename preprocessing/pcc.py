import numpy as np

from load_data import load_raw_data

PATH = '../data/'
RAW_FILE = 'diabetic_data_initial.csv'
TEST_RAW_FILE = 'test_diabetic_data_initial.csv'
UNFOLDED_SAVE_FILE = 'unfolded_feature_save.csv'

def get_pcc_for_all_numerical_features(features, labels, mappings=None):
    labels[labels == 'NO'] = 0
    labels[labels == '>30'] = 1
    labels[labels == '<30'] = 1

    for i in range(np.size(features, 1)):
        all_records = features[:,i]
        try:
            all_records_numerical = map(int, all_records)
            if mappings:
                feature_name = mappings[i]
            print '{}\t{}'.format(feature_name, np.corrcoef(all_records_numerical, labels)[0, 1])
        except:
            pass

f, l, mappings = load_raw_data(PATH + RAW_FILE, unfold=True)
# reverse the mappings
rev_mappings = {}
for m in mappings:
    rev_mappings[mappings[m]] = m
get_pcc_for_all_numerical_features(f, l, rev_mappings)
