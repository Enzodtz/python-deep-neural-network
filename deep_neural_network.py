import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class DeepNeuralNetwork:
    def __init__(self):
        pass

    def initialize_parameters(self):
        self.parameters = {}

        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]
            ) / np.sqrt(
                self.layer_dims[l - 1]
            )  # *0.01
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert self.parameters["W" + str(l)].shape == (
                self.layer_dims[l],
                self.layer_dims[l - 1],
            )
            assert self.parameters["b" + str(l)].shape == (self.layer_dims[l], 1)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

        return dZ

    def sigmoid_backward(self, dA, Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        return dZ

    def feed_forward_layer(self, A_prev, W, b, activation):
        Z = W.dot(A_prev) + b
        linear_cache = (A_prev, W, b)
        activation_cache = Z

        if activation == "sigmoid":
            A = self.sigmoid(Z)

        elif activation == "relu":
            A = self.relu(Z)

        cache = (linear_cache, activation_cache)

        self.caches.append(cache)

        return A

    def feed_forward(self, X):
        self.caches = []
        A = X

        for l in range(1, self.L):
            A_prev = A
            A = self.feed_forward_layer(
                A_prev,
                self.parameters["W" + str(l)],
                self.parameters["b" + str(l)],
                activation="relu",
            )

        AL = self.feed_forward_layer(
            A,
            self.parameters["W" + str(self.L)],
            self.parameters["b" + str(self.L)],
            activation="sigmoid",
        )

        return AL

    def compute_cost(self, AL, Y):
        # Compute loss from aL and y.
        cost = (1.0 / self.m) * (
            -np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T)
        )

        cost = np.squeeze(
            cost
        )  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost

    def backpropagate_z(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1.0 / m * np.dot(dZ, A_prev.T)
        db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def backpropagate_layer(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            return self.backpropagate_z(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            return self.backpropagate_z(dZ, linear_cache)

    def backpropagate(self, AL, Y):
        self.grads = {}
        Y = Y.reshape(AL.shape)

        # Initializing the backpropagation
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = self.caches[self.L - 1]
        (
            self.grads["dA" + str(self.L - 1)],
            self.grads["dW" + str(self.L)],
            self.grads["db" + str(self.L)],
        ) = self.backpropagate_layer(dAL, current_cache, activation="sigmoid")

        for l in reversed(range(self.L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backpropagate_layer(
                self.grads["dA" + str(l + 1)], current_cache, activation="relu"
            )
            self.grads["dA" + str(l)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp

    def update_parameters(self):
        # Update rule for each parameter. Use a for loop.
        for l in range(self.L):
            self.parameters["W" + str(l + 1)] = (
                self.parameters["W" + str(l + 1)]
                - self.learning_rate * self.grads["dW" + str(l + 1)]
            )
            self.parameters["b" + str(l + 1)] = (
                self.parameters["b" + str(l + 1)]
                - self.learning_rate * self.grads["db" + str(l + 1)]
            )

    def train(
        self,
        X,
        Y,
        learning_rate=0.0075,
        num_iterations=3000,
        print_cost=False,
        layer_dims=None,
    ):
        # Saving parameters
        self.m = Y.shape[1]
        self.L = len(layer_dims) - 1
        self.learning_rate = learning_rate

        # Initializing layer dims
        if not hasattr(self, "layer_dims"):
            assert layer_dims
            self.layer_dims = layer_dims
        else:
            assert layer_dims == None or layer_dims == self.layer_dims

        # Initializing parameters
        if not hasattr(self, "parameters"):
            self.initialize_parameters()

        # Initializing costs
        if not hasattr(self, "costs"):
            self.costs = []

        for i in tqdm(range(0, num_iterations)):
            AL = self.feed_forward(X)

            # Compute cost.
            cost = self.compute_cost(AL, Y)

            # Backward propagation.
            self.backpropagate(AL, Y)

            # Update parameters.
            self.update_parameters()

            # Print the cost every 100 training example
            if i % 100 == 0:
                self.costs.append(cost)

        return self.costs

    def predict(self, X):
        m = X.shape[1] or 1
        prediction = np.zeros((1, m))

        # Forward propagation
        probability = self.feed_forward(X)

        # convert probas to 0/1 predictions
        for i in range(0, probability.shape[1]):
            if probability[0, i] > 0.5:
                prediction[0, i] = 1
            else:
                prediction[0, i] = 0

        return prediction

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
