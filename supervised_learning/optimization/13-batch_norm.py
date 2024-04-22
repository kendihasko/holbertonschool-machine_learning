#!/usr/bin/env python3
'''
Batch Normalization
'''

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''
    Method that normalizes an unactivated output of a NN
    using batch normalization
    '''

    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    std_dev = np.sqrt(variance + epsilon)

    Z_norm = (Z - mean) / std_dev

    scaled = gamma * Z_norm + beta

    return scaled
