#!/usr/bin/env python3
'''
Test neural network
'''

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''
    A function that tests a neural network
    '''
    return network.evaluate(x=data,
                            y=labels,
                            verbose=verbose)
