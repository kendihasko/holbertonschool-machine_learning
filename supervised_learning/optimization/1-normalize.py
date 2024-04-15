#!/usr/bin/env python3
'''
Normalization/Standartization a matrix
'''

import numpy as np


def normalize(X, m, s):
    '''
    A function that normalizes (standardizes) a matrix
    '''
    Z = (X - m) / s
    return Z
