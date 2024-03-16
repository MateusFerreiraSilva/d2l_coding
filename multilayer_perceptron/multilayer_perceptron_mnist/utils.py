import numpy as np

import constants

def get_label_as_array(label):
    arr = np.zeros(constants.OUTPUT_LAYER_SIZE, dtype=float)
    arr[label] = 1.0

    return arr

def mse_loss(predicted, expected):
    return (predicted - expected) ** 2 / 2.0

def mse_loss_derivative(predicted, expected):
    return predicted - expected

def relu(z):
    return np.maximum(0.0, z)

def relu_derivative(z):
    return np.where(z > 0.0, 1.0, 0.0)

def softmax(z):
    exp_values = np.exp(z - np.max(z, axis=-1, keepdims=True))
    
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

def softmax_derivative(z):
    s = softmax(z)
    # Create an identity matrix with the same shape as z
    identity_matrix = np.eye(s.shape[1])
    # Compute the Jacobian matrix
    jacobian = s[:, :, np.newaxis] * (identity_matrix - s[:, np.newaxis, :])

    return jacobian