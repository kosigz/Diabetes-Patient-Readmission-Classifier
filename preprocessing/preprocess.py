import numpy as np

from load_data import load_raw_data

# ------------------------------------------------------------------------------
# Mostly done for my own curiosity and analysis.
def label_counts():
    f, l = load_raw_data(PATH + RAW_FILE)
    print '{}\t{}'.format('Label', 'Count')
    for label in np.unique(l):
        print '{}\t{}'.format(label, (l == label).sum())
