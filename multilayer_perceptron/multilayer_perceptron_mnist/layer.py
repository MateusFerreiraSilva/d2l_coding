import numpy as np
import cupy

import utils
import constants

class Layer:
    def __init__(self, input_size, output_size, is_first_layer=False, is_last_layer=False, learning_rate=constants.LEARNING_RATE):
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.zeros((1, output_size), dtype=float)
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.learning_rate = learning_rate

        self.z = None
        self.a = None

    def forward(self, x):
        self.z = cupy.dot(x, self.weights) + self.biases
        if self.is_last_layer:
            self.a = utils.softmax(self.z)
        else:
            self.a = utils.relu(self.z)

        return self.a

    def backward(self, x, y, previous_layer=None, successor_layer=None):
        if self.is_last_layer:
            # self.d_a = utils.categorical_crossentropy_derivative(self.a, y)
            # self.d_z = self.d_a * utils.softmax_derivative(self.z.T)

            self.d_z = utils.delta_cross_entropy(self.a, y)
        else:
            self.d_a = successor_layer.d_z.dot(successor_layer.weights.T)
            self.d_z = self.d_a * utils.relu_derivative(self.z)

        self.d_b = cupy.sum(self.d_z, axis=0)

        if self.is_first_layer:
            self.d_w = cupy.dot(x.T, self.d_z)
        else:
            self.d_w = cupy.dot(previous_layer.a.T, self.d_z)
    
    def update_weights_and_biases(self):
        self.weights -= self.learning_rate * self.d_w
        self.biases -= self.learning_rate * self.d_b

    # def backward(self, layer_input, expected_output, successor_layer_output=None):
    #     if self.is_last_layer:
    #         d_a = utils.mse_loss_derivative(self.a, expected_output)
    #         d_z = np.dot(d_a, utils.softmax_derivative(self.z))
    #     else:
    #         d_a = np.dot(self.weights, successor_layer_output)
    #         d_z = np.dot(d_a, utils.relu_derivative(d_a))

    #     d_b = d_z
    #     d_w = np.dot(d_z.T, layer_input)

    #     return d_a, d_z, d_b, d_w