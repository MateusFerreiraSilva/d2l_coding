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

        return layer_input # last layer output

    def backpropagation(self, x, y):
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            previous_layer = None if layer.is_first_layer else self.layers[i - 1]
            successor_layer = None if layer.is_last_layer else self.layers[i + 1]

            layer.backward(x, y, previous_layer=previous_layer, successor_layer=successor_layer)

    def update_weights_and_biases(self):
        for layer in self.layers:
            layer.update_weights_and_biases()

    def fit(self, x, y):
        predicted = self.forward(x)

        self.backpropagation(x, y)

        self.update_weights_and_biases()

        loss = utils.categorical_crossentropy(predicted, y)

        return loss

    def train(self, train_imgs, train_labels, epochs=constants.EPOCHS):
        print('Training...\n')
        total_loss = None
        for i in range(epochs):
            total_loss = 0.0
            for img, label in zip(train_imgs, train_labels):
                vectorized_label = utils.vectorize_label(label)
                loss = self.fit(img, vectorized_label)
                total_loss += loss
            print(f'[epoch: {i + 1}] [loss: {total_loss}]')

        print('Training finished!\n')

