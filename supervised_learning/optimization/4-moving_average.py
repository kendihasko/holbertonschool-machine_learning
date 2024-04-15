#!/usr/bin/env python3
'''
Calculates the weighted moving average of a data set
'''

import numpy as np


def moving_average(data, beta): 
    '''
    A function that calculates the weighted moving average of a data set
    '''
    m_av=[]

    w = 0

    for i, d in enumerate(data):
        w = beta * w + (1+beta) * d
        w_new = w/(i-beta**(i+1))
        m_av.append(w_new)

    return m_av
