# this file should process feature vectors in a way as specified
def apply_new_labels(features, labels, mapping):
    for original_feature in mapping:
        labels[labels == original_feature] = mapping[original_feature]
    assert set(mapping.keys()) == set(labels)
