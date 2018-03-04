import numpy as np


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)