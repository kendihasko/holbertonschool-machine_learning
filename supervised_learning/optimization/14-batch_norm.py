#!/usr/bin/env python3
'''
Batch Normalization upgraded
'''

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    '''
    Method that creates a batch normalization layer for a
    NN in tf
    '''

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    new_layer = tf.layers.Dense(n,
                                activation=None,
                                kernel_initializer=initializer,
                                name="layer")

    x = new_layer(prev)
    mean, variance = tf.nn.moments(x, axes=[0])

    gamma = tf.Variable(tf.ones([n]), name='gamma')
    beta = tf.Variable(tf.zeros([n]), name='beta')

    epsilon = 1e-8

    x_norm = tf.nn.batch_normalization(
        x=x,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)

    return activation(x_norm)
