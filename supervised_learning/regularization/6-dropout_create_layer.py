#!/usr/bin/env python3
"""
    Create layer with Dropout regularization
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        function that creates layer of NN using dropout
    """
    dropout_layer = tf.compat.v1.layers.Dropout(rate=keep_prob)

    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')

    new_layer = (
        tf.layers.Dense(n,
                        activation=activation,
                        kernel_initializer=initializer,
                        kernel_regularizer=dropout_layer,
                        name="layer"))

    output = new_layer(prev)

    return output
