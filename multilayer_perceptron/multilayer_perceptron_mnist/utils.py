import numpy as np

import constants

def vectorize_label(label):
    vector = np.zeros(constants.OUTPUT_LAYER_SIZE, dtype=float)
    vector[label] = 1.0

    return vector

def mse_loss(predicted, expected):
    return (predicted - expected) ** 2 / 2.0 # should i sum all?

def mse_loss_derivative(predicted, expected):
    return predicted - expected

def categorical_crossentropy(y_pred, y):
    # Clip y_pred to avoid log(0) and ensure numerical stability
    y_pred = np.clip(y_pred, constants.EPSILON, 1 - constants.EPSILON)
    # Compute the cross-entropy loss
    loss = -np.sum(y * np.log(y_pred))

    return loss

# def categorical_crossentropy_derivative(y_pred, y):
#     # Clip y_pred to avoid log(0) and ensure numerical stability
#     y_pred = np.clip(y_pred, constants.EPSILON, 1 - constants.EPSILON)
#     # Compute the derivative of the cross-entropy loss
#     derivative = y_pred - y
    
#     return derivative

def delta_cross_entropy(y_pred, y):
   grad = softmax(y_pred)
   grad[0, y.argmax()] -= 1
   
   return grad

def relu(z):
    return np.maximum(0.0, z)

def relu_derivative(z):
    return np.where(z > 0.0, 1.0, 0.0)

def naive_softmax(z):
    s = np.exp(z)
    return s / s.sum()


def softmax(z):
    return naive_softmax(z - max(z))

# def softmax_derivative(z):
#     s = softmax(z)
#     # Create an identity matrix with the same shape as z
#     identity_matrix = np.eye(s.shape[1])
#     # Compute the Jacobian matrix
#     jacobian = s[:, :, np.newaxis] * (identity_matrix - s[:, np.newaxis, :])

#     return jacobian

# def softmax_derivative(z):
#     s = softmax(z)
#     jacobian = np.diag(s) - np.outer(s, s)

#     return jacobian