import numpy as np
from scipy.special import expit


def sigmoid(x: np.ndarray):
    return expit(x)


def sigmoid_prime(x: np.ndarray):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)