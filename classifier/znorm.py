# decorator to z-normalize the input data of a functions
# input function arguments: X, Y
# output function arguments: X, Y, normalize
def znorm_dec(fn):
    def znorm_fn(X, Y):
        znorm_X, normalize = znorm(X)
        return fn(znorm_X, Y, normalize=normalize)
    return znorm_fn

# perform z-score normalization on the dataset, providing the normalization
# function to be used on test points
def znorm(X):
    stdev = X.std(axis=0)
    stdev[stdev == 0] = 1 # replace 0s with 1s for division
    mean = X.mean(axis=0)

    normalize = lambda x: (x - mean) / stdev

    return normalize(X), normalize
