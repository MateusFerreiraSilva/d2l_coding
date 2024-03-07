# https://abtinmy.github.io/CS-SBU-NeuralNetwork/lectures/introduction/MLP-Scratch-Iris

import numpy as np
from random import randint
from sklearn import datasets

EPOCHS=1000
IRIS_DATASET_SIZE = 150

def get_random_interval(num):
    interval_start = randint(0, IRIS_DATASET_SIZE - num)
        
    return [interval_start, interval_start + num - 1]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)

        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))


    def fit(self, X, y, epochs=EPOCHS):
        # feedforward
        layer1 = X.dot(self.weights1) + self.bias1
        activation1 = sigmoid(layer1)
        layer2 = activation1.dot(self.weights2) + self.bias2
        activation2 = sigmoid(layer2)
  
        # backpropagation
        error = activation2 - y # y = expected output
        d_weights2 = activation1.T.dot(error * sigmoid_derivative(layer2))
        d_bias2 = np.sum(error * sigmoid_derivative(layer2), axis=0, keepdims=True)
        error_hidden = error.dot(self.weights2.T) * sigmoid_derivative(layer1)
        d_weights1 = X.T.dot(error_hidden)
        d_bias1 = np.sum(error_hidden, axis=0, keepdims=True)

        # update weights and biases
        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1

    def train(self):
        iris_dataset = datasets.load_iris()
        X = iris_dataset["data"][:, (2, 3)] # 150 examples, take 2 features petal length, petal width
        y = (iris_dataset["target"] == 2).astype(int)  # 1 if Iris-Virginica, else 0
        y =  y.reshape([IRIS_DATASET_SIZE, 1])
        
        print("training...")
        self.fit(X, y)
        print("Done!\n")

    def predict(self, X):
        layer1 = X.dot(self.weights1) + self.bias1
        activation1 = sigmoid(layer1)
        layer2 = activation1.dot(self.weights2) + self.bias2
        activation2 = sigmoid(layer2)

        return (activation2 > 0.5).astype(int)


mlp = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1)
mlp.train()


rand_interval = get_random_interval(10)
iris_dataset = datasets.load_iris()
X = iris_dataset["data"][:, (2, 3)]
y = (iris_dataset["target"] == 2).astype(int)
y =  y.reshape([IRIS_DATASET_SIZE, 1])
y_pred = mlp.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")