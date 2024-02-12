#!/usr/bin/env python3
"""
A script that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    '''
    A function that calculates the integral of a polynomial
    '''
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, int):
            return None

    while poly and poly[-1] == 0:
        poly.pop()

    if not poly:
        return [C]

    integral_coeffs = [poly[i] / (i + 1) for i in range(len(poly))]

    integral_coeffs.insert(0, C)

    return int(integral_coeffs)
