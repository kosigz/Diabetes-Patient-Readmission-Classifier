from preprocessing.preprocess import get_preprocessed_data as get_data
from classifier.SVMClassifier import SVMClassifier
from classifier.test_classifier import test_classifier_accuracy_with_num_records

def test_svm():
    # hangs on the larger values, i.e. 10 and 25, so I am taking those out here
    for C in (0.01, 0.1, 1, 2, 5):
        for kernel in ("linear", "poly", "rbf", "sigmoid"):
            if kernel == "poly":
                for degree in range(2, 4):
                    test_classifier_accuracy_with_num_records(SVMClassifier(C, kernel=kernel, degree=degree, verbose=True), 1000)
            else:
                test_classifier_accuracy_with_num_records(SVMClassifier(C, kernel=kernel), 1000)

test_svm()
print 'Done.'
