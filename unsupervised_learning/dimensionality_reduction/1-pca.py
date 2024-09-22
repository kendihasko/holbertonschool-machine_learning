#!/usr/bin/env python3
"""
Defines function that performs principal components analysis (PCA) on dataset
"""


import numpy as np


def pca(X, ndim):
    """
    Performs principal components analysis (PCA) on a dataset
    """
    # n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    u, s, v = np.linalg.svd(A)
    W = v.T[:, :ndim]
    T = np.matmul(A, W)
    return (T)
