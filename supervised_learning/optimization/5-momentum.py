#!/usr/bin/env python3
'''
Updates a variable using the gradient descent
with momentum optimization algorithm
'''

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
    A function that updates a variable
    using the gradient descent with momentum optimization algorithm
    '''
    
