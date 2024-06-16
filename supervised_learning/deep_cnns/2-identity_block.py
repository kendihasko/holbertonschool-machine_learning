#!/usr/bin/env python3
"""
Defines a function that builds an identity block using Keras
"""

from tensorflow import keras as K

def identity_block(A_prev, filters):
    """
    Builds an identity block using Keras
    """
    
    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    activation = K.activations.relu

    # First component of the main path
    C11 = K.layers.Conv2D(filters=F11,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=init)(A_prev)
    Batch_Norm11 = K.layers.BatchNormalization(axis=3)(C11)
    ReLU11 = K.layers.Activation(activation)(Batch_Norm11)

    # Second component of the main path
    C3 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=init)(ReLU11)
    Batch_Norm3 = K.layers.BatchNormalization(axis=3)(C3)
    ReLU3 = K.layers.Activation(activation)(Batch_Norm3)

    # Third component of the main path
    C12 = K.layers.Conv2D(filters=F12,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=init)(ReLU3)
    Batch_Norm12 = K.layers.BatchNormalization(axis=3)(C12)

    # Add shortcut value to the main path, and pass it through a RELU activation
    Addition = K.layers.Add()([Batch_Norm12, A_prev])
    output = K.layers.Activation(activation)(Addition)

    return output
