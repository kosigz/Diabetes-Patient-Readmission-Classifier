from preprocessing.preprocess import get_preprocessed_data as get_data
from classifier.test_classifier import test_classifier_accuracy_with_num_records
from . import ranker
from classifier.KNNClassifier import KNNClassifier


fts, labels = get_data('diabetic_data_initial.csv', nrows=5000, unfold=True)

corr = ranker.get_correlation(fts, labels).abs().sort_values(ascending=False)

def test_filter_accuracy_knn():
    prev_acc = 0
    for i in range(1, len(corr), 10):
        count = 0
        features = []
        for n, v in corr.iteritems():
            count += 1
            if count == i:
                break
            if n != 'payer_code':
                features.append(n)
        for k in range(1, 4):
            acc = test_classifier_accuracy_with_num_records(KNNClassifier(k, n_jobs=4), features_to_use=features)
            print '{}\t{}\t{}'.format(k, len(features), acc)

test_filter_accuracy_knn()
