#!/usr/bin/env python3
"""
A script that concatenates two matrices along a specific axis
"""

import numpy as np
def np_cat(mat1, mat2, axis=0):
    """
A function that concatenates two matrices along a specific axis
    """
    np.concatenate((mat1, mat2), axis = axis)