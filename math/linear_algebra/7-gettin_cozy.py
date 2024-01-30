#!/usr/bin/env python3
"""
...
"""

def cat_matrices2D(mat1, mat2, axis=0): 
    """
A function that concatenates two matrices along a specific axis
    """

    if axis == 0:
        if len(mat1[0] != len(mat2[0])):
            return None

            