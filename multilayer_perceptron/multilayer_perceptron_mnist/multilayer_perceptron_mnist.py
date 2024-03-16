import numpy as np

import constants
import utils
from layer import Layer

class MultilayerPerceptron:
    def __init__(self):
        self.layers = [
            Layer(constants.INPUT_LAYER_SIZE, constants.HIDDEN_LAYER_SIZE, is_first_layer=True),
            Layer(constants.HIDDEN_LAYER_SIZE, constants.HIDDEN_LAYER_SIZE),
            Layer(constants.HIDDEN_LAYER_SIZE, constants.OUTPUT_LAYER_SIZE, is_last_layer=True)
        ]

    def forward(self, x):
        layer_input = x
        for layer in self.layers:
            layer_input = layer.forward(layer_input)

    def backpropagation(self, x, y):
        pass

    def update_weights_and_biases(self, gradients):
        pass

    def fit(self, x, y):
        predicted = self.forward(x)

        gradients = self.backpropagation(x, y)

        self.update_weights_and_biases(gradients)

        loss = utils.mse_loss(predicted, y)

        return loss

    def train(self, train_imgs, train_labels, epochs=constants.EPOCHS):
        print('Training...\n')
        total_loss = None
        for i in range(epochs):
            total_loss = 0.0
            for img, label in zip(train_imgs, train_labels):
                label_arr = utils.get_label_as_array(label)
                loss = self.fit(img, label_arr)
                total_loss += loss
            print(f'[epoch: {i + 1}] [loss: {total_loss}]')

        print('Training finished!\n')

