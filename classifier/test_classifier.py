def gen_clusters(*clusters):
    import numpy as np

    data, outputs = np.empty(0), np.empty(0)
    for i, (mean, stdev, n) in enumerate(clusters):
        data = np.append(data, np.random.normal(mean, stdev, n))
        outputs = np.append(outputs, [i] * n)

    return data[:,None], outputs


def test_classifier_accuracy(classifier, n=1000):
    m = n / 100

    train_X, train_Y = gen_clusters((0, 0.5, n), (1, 0.25, n), (2, 1, n))
    test_X, test_Y = gen_clusters((0, 0.5, m), (1, 0.25, m), (2, 1, m))

    classifier.train(train_X, train_Y)
    return classifier.accuracy(test_X, test_Y)

def test_classifier(classifier, *args, **kwargs):
    print "{classifier} achieved {accuracy:2.2f}% accuracy on generated test data".format(
        classifier=classifier,
        accuracy=100 * test_classifier_accuracy(classifier, *args, **kwargs))