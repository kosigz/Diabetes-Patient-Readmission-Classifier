from sklearn import datasets
from sklearn.cross_validation import train_test_split

# import some test data
#ds = datasets.load_iris()
#X, Y = ds.data, ds.target
X, Y = datasets.make_blobs(n_samples=1000, n_features=4, cluster_std=7, random_state=1)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=1)

def gen_clusters(*clusters):
    import numpy as np

    data, outputs = np.empty(0), np.empty(0)
    for i, (mean, stdev, n) in enumerate(clusters):
        data = np.append(data, np.random.normal(mean, stdev, n))
        outputs = np.append(outputs, [i] * n)

    return data[:,None], outputs


def test_classifier_accuracy(classifier, n=1000):
    m = n / 100

    classifier.train(train_X, train_Y)
    return classifier.accuracy(test_X, test_Y)

def test_classifier(classifier, *args, **kwargs):
    print "{classifier} achieved {accuracy:2.2f}% accuracy on generated test data".format(
        classifier=classifier,
        accuracy=100 * test_classifier_accuracy(classifier, *args, **kwargs))