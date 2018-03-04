import pandas as pd
from sklearn.model_selection import train_test_split


def get_splits():
    X = pd.read_csv('../data/train_x.csv', sep=',')
    y = pd.read_csv('../data/train_y.csv', sep=',')

    X = X.as_matrix().reshape(-1, 64, 64)
    y = y.as_matrix().reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    return X_train, X_test, y_train, y_test
