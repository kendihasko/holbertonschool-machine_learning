#!/usr/bin/env python3
'''
Updates a variable using the gradient descent
with momentum optimization algorithm
'''


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
    A function that updates a variable
    using the gradient descent with momentum optimization algorithm
    '''
    dW = beta1 * v + (1 - beta1) * grad

    var_new = var - dW * alpha

    return var_new, dW
