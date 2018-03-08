import numpy as np
from scipy.special import expit


def sigmoid(x: np.ndarray):
    return expit(x)


def sigmoid_prime(x: np.ndarray):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)


class CrossEntropyLoss:

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a - y