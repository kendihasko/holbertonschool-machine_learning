#!/usr/bin/env python3
'''
Save and load weight function
'''

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    '''
    A function that saves a model's weights
    '''
    network.save_weights(filepath=filename,
                         save_format=save_format)


def load_weights(network, filename):
    '''
    A function that loads a model's weights
    '''
    network.load_weights(filepath=filename)
