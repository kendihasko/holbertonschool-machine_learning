#!/usr/bin/env python3
'''
Adam upgraded
'''

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''
    Method that updates variable in place using
    Adam optimizer algo
    '''

    new_v = beta1 * v + (1 - beta1) * grad
    new_s = beta2 * s + (1 - beta2) * grad**2

    v_corrected = new_v / (1 - beta1**t)
    s_corrected = new_s / (1 - beta2**t)

    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, new_v, new_s
