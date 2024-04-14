#!/usr/bin/env python3
'''
Save and load model function
'''

import tensorflow.keras as K


def save_model(network, filename):
    '''
        A function that saves an entire model
    '''
    network.save(filename)


def load_model(filename):
    '''
        A function that loads an entire model
    '''
    return K.models.load_model(filename)
