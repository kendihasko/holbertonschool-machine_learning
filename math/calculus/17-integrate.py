#!/usr/bin/env python3
"""
A script that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    A function that calculates the integral of a polynomial
    """
    if not isinstance(poly, list) or \
            not all(isinstance(coeff, (int, float)) for coeff in poly) or \
            C is not None and not isinstance(C, int):
        return None

    if not poly:
        return None

    while poly and poly[-1] == 0:
        poly.pop()

    if not poly:
        return [C]

    integral_coeffs = [0] * (len(poly) + 1)
    integral_coeffs[0] = C
    for i in range(len(poly)):
        integral_coeffs[i + 1] = poly[i] / (i + 1)
        if integral_coeffs[i + 1].is_integer():
            integral_coeffs[i + 1] = int(integral_coeffs[i + 1])

    return integral_coeffs
