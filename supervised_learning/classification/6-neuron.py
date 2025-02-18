#!/usr/bin/env python3
'''
Neuron class that defines a single neuron performing binary classification
'''


import numpy as np


class Neuron:
    '''
    A single neuron performing binary classification
    '''

    def __init__(self, nx):
        '''
        Class constructor
        '''
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
        Get method for property Weights
        '''
        return self.__W

    @property
    def b(self):
        '''
        Get method for property bias
        '''
        return self.__b

    @property
    def A(self):
        '''
        Get method for property prediction/output/activation function
        '''
        return self.__A

    def forward_prop(self, X):
        '''
        Calculates the forward propagation of the neuron,
        neuron should use a sigmoid activation function
        '''
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1+np.exp(-z))
        return self.A

    def cost(self, Y, A):
        '''
        Calculates the cost of the model using logistic regression
        '''
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1/m) * (-(m_loss))
        return cost

    def evaluate(self, X, Y):
        '''
        Evaluates a neuron's prediction
        '''
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
        Calculates one pass of gradient descent on the neuron
        '''
        m = Y.shape[1]
        dz = A - Y
        d__W = (1 / m) * (np.matmul(X, dz.transpose())).transpose()
        d__b = (1 / m) * (np.sum(dz))

        self.__W = self.W - (alpha * d__W)
        self.__b = self.b - (alpha * d__b)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''
        Trains the neuron
        '''
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for itr in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
