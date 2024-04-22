#!/usr/bin/env python3
'''
Learning Rate Decay
'''

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''
    Method to updates the learning rate using
    inverse time decay in numpy
    '''

    factor = (1 + decay_rate * (global_step//decay_step))

    alpha = alpha / factor
    return alpha
