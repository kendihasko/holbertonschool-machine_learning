#!/usr/bin/env python3
"""
    Gradient Descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        function that updates the weights and biases of NN
        using gradient descent with L2 regularization
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        L2_regularization = lambtha / m * weights['W' + str(layer)]

        A_prev = cache['A' + str(layer - 1)]

        dW = np.matmul(dZ, A_prev.T) / m + L2_regularization
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(weights['W' + str(layer)].T, dZ)

        if layer != 1:
            dZ = dA * (1 - A_prev ** 2)
        else:
            dZ = dA

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
