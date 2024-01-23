#!/usr/bin/env python3
"""
A function def matrix_shape(matrix): that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
A function def matrix_shape(matrix): that calculates the shape of a matrix
    """

    matrix_shape = []
    while type(matrix) is list:
        matrix_shape.append(len(matrix))
        matrix = matrix[0]

    return matrix_shape
