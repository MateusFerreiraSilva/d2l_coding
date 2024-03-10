import random
import numpy as np

BATCH = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class MultilayerPerceptron:
    """

    Simple implementation of a Multilayer Perceptron (MLP) that learns the logic operation AND.

    This MLP consists of an input layer with two neurons (x1 and x2), a single hidden layer with two neurons,
    and an output layer with one neuron. It is trained to perform the logic operation AND on binary inputs.

    Attributes:
    - learning_rate (float): The learning rate used during weight updates.
    - w11, w12, w21, w22, w3, w4 (float): Weights associated with the connections between neurons.
    - b1, b2, b3 (float): Biases associated with the neurons.

    Methods:
    - forward(X, y): Performs forward propagation given input values X and target output y.
    - backpropagation(X, y): Performs backpropagation to update weights and biases.
    - fit(X, y): Fits the model to the provided input-output pair (X, y).
    - train(batch, epochs): Trains the model on a batch of input-output pairs for a specified number of epochs.
    - prediction(data): Predicts the output for a given input data and prints the prediction and loss.

    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

        self.w11 = random.uniform(0, 1)
        self.w12 = random.uniform(0, 1)
        self.w21 = random.uniform(0, 1)
        self.w22 = random.uniform(0, 1)
        self.w3 = random.uniform(0, 1)
        self.w4 = random.uniform(0, 1)

        self.b1 = 0.0
        self.b2 = 0.0
        self.b3 = 0.0

    def forward(self, X, y):
        self.x1, self.x2 = X

        self.z1 = self.x1 * self.w11 + self.x2 * self.w12 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.x1 * self.w21 + self.x2 * self.w22 + self.b2
        self.a2 = sigmoid(self.z2)

        self.z3 = self.a1 * self.w3 + self.a2 * self.w4 + self.b3
        self.o1 = sigmoid(self.z3)

        n = 1 # in this case always will be 1

        error = 1 / 2 * 1 / n * (self.o1 - y) ** 2
        prediction = self.o1

        return (prediction, error)
    
    def backpropagation(self, y):
        self.d_o1 = self.o1 - y # d_error / d_o1
        self.d_z3 = self.d_o1 * sigmoid_derivative(self.z3)
        self.d_b3 = self.d_z3 * 1
        
        self.d_w3 = self.d_z3 * self.a1
        self.d_w4 = self.d_z3 * self.a2

        self.d_a2 = self.d_z3 * self.w4
        self.d_z2 = self.d_a2 * sigmoid_derivative(self.z2)
        self.d_w21 = self.d_z2 * self.x1
        self.d_w22 = self.d_z2 * self.x2
        self.d_b2 = self.d_z2 * 1

        self.d_a1 = self.d_z3 * self.w3
        self.d_z1 = self.d_a1 * sigmoid_derivative(self.z1)
        self.d_w11 = self.d_z1 * self.x1
        self.d_w12 = self.d_z1 * self.x2
        self.d_b1 = self.d_z1 * 1

    def update_weights_and_biases(self):
        self.w11 -= self.learning_rate * self.d_w11
        self.w12 -= self.learning_rate * self.d_w12
        self.w21 -= self.learning_rate * self.d_w21
        self.w22 -= self.learning_rate * self.d_w22
        self.w3 -= self.learning_rate * self.d_w3
        self.w4 -= self.learning_rate * self.d_w4

        self.b1 -= self.learning_rate * self.d_b1
        self.b2 -= self.learning_rate * self.d_b2
        self.b3 -= self.learning_rate * self.d_b3

    def fit(self, X, y):
        self.forward(X, y)

        self.backpropagation(y)

        self.update_weights_and_biases()

    def train(self, batch, epochs=10000):
        for i in range(epochs):
            for b in batch:
                X, y = b
                self.fit(X, y)

    def prediction(self, data):
        X, y = data

        prediction, loss = self.forward(X, y)

        print(f'prediction: {prediction > 0.5}')
        print(f'loss: {loss}')

# main
        
model = MultilayerPerceptron()
model.train(BATCH)

for b in BATCH:
    model.prediction(b)