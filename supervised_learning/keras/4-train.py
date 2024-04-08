#!/usr/bin/env python3

'''
Trains a model using mini-batch gradient descent
'''

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    '''
    A function that trains a model using mini-batch gradient descent
    '''