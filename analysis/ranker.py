from preprocessing.preprocess import get_preprocessed_data as get_data

def get_correlation(fts, labels):
    return fts.corrwith(labels).sort_values(ascending=False)

fts, labels = get_data('diabetic_data_initial.csv')
# print get_correlation(fts, labels)
