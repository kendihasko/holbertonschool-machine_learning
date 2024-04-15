#!/usr/bin/env python3
'''
Updates a variable using the RMSProp optimization algorithm
'''


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''
    A function that updates a variable
    using the RMSProp optimization algorithm
    '''
    squared_gradient = beta2 * s + (1 - beta2) * (grad**2)
    updated_var = var - (alpha * grad) / (np.sqrt(squared_gradient) + epsilon)
    return updated_var, squared_gradient
