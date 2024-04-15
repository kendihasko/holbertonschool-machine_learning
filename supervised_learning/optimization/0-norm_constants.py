#!/usr/bin/env python3
'''
Normalization/Standartization
'''

import numpy as np


def normalization_constants(X):
    '''
    A function that calculates the normalization 
    (standardization) constants of a matrix
    '''
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    return mean, std