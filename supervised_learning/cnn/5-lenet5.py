#!/usr/bin/env python3
'''
    LeNet-5 (Keras)
'''

import tensorflow.keras as K


def lenet5(X):
    '''
        A function that builds a modified version of the LeNet-5
        network architecture using keras
    '''

    # intializer
    initializer = K.initializers.HeNormal()

    model = K.Sequential([
        K.layers.Conv2D(filters=6,
                        kernel_size=5,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='relu'),
        K.layers.MaxPooling2D(pool_size=2,
                              strides=2),
        K.layers.Conv2D(filters=16,
                        kernel_size=5,
                        padding='valid',
                        kernel_initializer=initializer,
                        activation='relu'),
        K.layers.MaxPooling2D(pool_size=2,
                              strides=2),
        K.layers.Flatten(),
        K.layers.Dense(120,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(84,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(10,
                       kernel_initializer=initializer,
                       activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
