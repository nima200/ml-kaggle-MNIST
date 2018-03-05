from src.math_utils import *
import random


class NNClassifier:
    def __init__(self, structure):
        self.n_layers = structure.shape[0]
        self.structure = structure
        # Initialize weights and biases to random values
        # Ignore first element of structure: n_input_neurons
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]
        # weights[i][j][k] denotes weight of connection between neuron k in layer i and j in layer i + 1
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(structure[:-1], structure[1:])]

    def _feed_forward(self, X):
        X_flat = X.flatten()
        X_flat = X_flat.reshape(X.size, 1)
        """Return the output of the network if ''X'' is input."""
        for bias, weight in zip(self.biases, self.weights):
            X_flat = sigmoid(weight.dot(X_flat) + bias)
        return X_flat

    def fit(self, training_data, epochs, mini_batch_size, l_rate, test_data, print_rate=10):
        if test_data is not None:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self._mini_batch_update(mini_batch, l_rate)
            if test_data is not None:
                print("Epoch %d: %d / %d" % (i, self.predict(test_data), n_test))
            else:
                print("Epoch %d complete" % i)

    def _mini_batch_update(self, mini_batch, l_rate):
        # Initiate weight and bias gradients to zero for each batch
        grad_w = [np.zeros(weight[1].shape) for weight in enumerate(self.weights)]
        grad_b = [np.zeros(bias[1].shape) for bias in enumerate(self.biases)]
        for x, y in mini_batch:
            # x_flat = x.flatten().reshape(x.size, 1)
            delta_grad_w, delta_grad_b = self._back_propagate(x, y)
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
        # Update Weights and Biases using gradient descent
        self.weights = [weight - (l_rate / len(mini_batch)) * gw
                        for weight, gw in zip(self.weights, grad_w)]
        self.biases = [bias - (l_rate / len(mini_batch)) * gb
                       for bias, gb in zip(self.biases, grad_b)]

    def _back_propagate(self, x, y):
        grad_w = [np.zeros(weight[1].shape) for weight in enumerate(self.weights)]
        grad_b = [np.zeros(bias[1].shape) for bias in enumerate(self.biases)]

        # FORWARD PASS
        # First (input) layer activation
        activation = x
        # List of all layers' activations
        activations = [x]
        outputs = []
        for b, W in zip(self.biases, self.weights):
            output = W.dot(activation) + b
            outputs.append(output)
            activation = sigmoid(output)
            activations.append(activation)

        # BACKWARD PASS
        delta = self._loss_derivative(activations[-1], y) * sigmoid_prime(outputs[-1])
        grad_w[-1] = delta.dot(activations[-2].T)
        grad_b[-1] = delta
        for layer in range(2, self.n_layers):
            output = outputs[-layer]
            sig_prime = sigmoid_prime(output)
            delta = self.weights[-layer + 1].T.dot(delta) * sig_prime
            grad_w[-layer] = delta.dot(activations[-layer - 1].T)
            grad_b[-layer] = delta

        return grad_w, grad_b

    def predict(self, test_data):
        test_results = [(np.argmax(self._feed_forward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    @staticmethod
    def _loss_derivative(output_activations, y):
        return output_activations - y