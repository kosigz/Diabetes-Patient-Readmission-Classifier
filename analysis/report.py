import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from preprocessing.preprocess import get_preprocessed_data as get_data
from classifier.SVMClassifier import SVMClassifier
from classifier.KNNClassifier import KNNClassifier
from classifier.LogisticRegressionClassifier import LogisticRegressionClassifier
from classifier.RandomForestClassifier import RandomForestClassifier
from classifier.test_classifier import test_classifier_accuracy#, test_classifier_accuracy_with_num_records



PHI = (1 + 5 ** 0.5) / 2
OUTPUT_PATH = "./output"

def save_plot_img(file_name):
    plt.savefig(join(OUTPUT_PATH, file_name), bbox_inches="tight")

def test_svm():
    # hangs on the larger values, i.e. 10 and 25, so I am taking those out here
    for C in [6, 7, 8]:
        for kernel in ("poly"):
            if kernel == "poly":
                for degree in range(3, 4):
                    print 'C: {}\nKernel: {}\nDegree: {}\nAccuracy: {}'.format(
                    C, kernel, degree,
                    test_classifier_accuracy_with_num_records(SVMClassifier(C, kernel=kernel, degree=degree)))
            else:
                print 'C: {}\nKernel: {}\nAccuracy: {}'.format(
                C, kernel,
                test_classifier_accuracy_with_num_records(SVMClassifier(C, kernel=kernel)))

def test_knn():
    for k in (1, 2, 4, 8, 16, 25, 32, 50, 64):
        print 'KNN with [k = {}] accuracy is [{}]'.format(k, test_classifier_accuracy_with_num_records(KNNClassifier(k, n_jobs=4)))

def test_log_reg():
    for C in [0.0001, 0.001, 0.01, 0.1]:
        for degree in range(3, 4):
            print 'C: {}\nDegree: {}\nAccuracy: {}'.format(
            C, degree,
            test_classifier_accuracy_with_num_records(LogisticRegressionClassifier(C=C, degree=degree)))

def test_random_forests():
    for t in (128, 256, 512, 1024):
        print 'Num Trees: {}\nAccuracy: {}'.format(
        t,
        test_classifier_accuracy_with_num_records(RandomForestClassifier(n=t)))

def learning_curve(classifier, sizes, classifier_name=None, **kwargs):
    classifier_name = classifier_name or str(classifier)
    acc = []
    for size in sizes:
        print "Size: ", size
        acc.append(
            100 * test_classifier_accuracy(classifier, num_samples=size, **kwargs))

    # build plot
    plt.figure(figsize=(5 * PHI, 5))
    plt.plot(sizes, acc, "bo")
    plt.ylim(ymax=100)

    # create title and axis labels
    plt.title("Learning Curve\n{}".format(classifier_name))
    plt.xlabel("Size of Training Set")
    plt.ylabel("Accuracy (%)")

    # display the plot
    # plt.show()
    # save the plot as an image
    save_plot_img("{}-learning-curve.png".format(classifier))



print learning_curve(RandomForestClassifier(3), range(100, 5001, 250))
