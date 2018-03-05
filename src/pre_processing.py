import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_splits():
    X = pd.read_csv('./data/train_x.csv', sep=',')
    y = pd.read_csv('./data/train_y.csv', sep=',')

    X = X.as_matrix().reshape(-1, 64, 64)
    y = y.as_matrix().reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25)
    X_train = [np.reshape(x, (x.size, 1)) for x in X_train]
    y_train = [vectorized_result(y) for y in y_train]
    training_data = list(zip(X_train, y_train))

    X_test = [np.reshape(x, (x.size, 1)) for x in X_test]
    test_data = list(zip(X_test, y_test))

    return training_data, test_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e