import numpy as np

import utils

class Layer:
    def __init__(self, input_size, output_size, is_first_layer=False, is_last_layer=False):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size), dtype=float)
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.z = None
        self.a = None

    def forward(self, x):
        self.z = np.dot(x, self.weights) + self.bias
        if self.is_last_layer:
            self.a = utils.softmax(self.z)
        else:
            self.a = utils.relu(self.z)

        return self.a

    def backward(self, x, y):
        pass