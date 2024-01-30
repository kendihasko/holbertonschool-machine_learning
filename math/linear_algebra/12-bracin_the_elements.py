#!/usr/bin/env python3
"""
A script that performs various operations
"""


def np_elementwise(mat1, mat2):
    """
A function that performs various operations
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return (add, sub, mul, div)
