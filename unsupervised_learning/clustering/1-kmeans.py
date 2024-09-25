#!/usr/bin/env python3
"""K means"""

import numpy as np

def kmeans(X, k, iterations=1000):
    """
    K-means on a data set
    """

    # if not isinstance(X, np.ndarray) or len(X.shape) != 2:
    #     return None, None
    # if not isinstance(k, int) or k <= 0:
    #     return None, None
    # if not isinstance(iterations, int) or iterations <= 0:
    #     return None, None

    # Setting min and max values per column
    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Centroid initialization
    C = np.random.uniform(X_min, X_max, size=(k, d))

    for i in range(iterations):
        # Calculate distances and assign clusters in one go
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_C = np.array([X[clss == c].mean(axis=0) if X[clss == c].size > 0 else np.random.uniform(X_min, X_max, d) for c in range(k)])

        # Check for convergence
        if np.all(C == new_C):
            break

        C = new_C

    return C, clss

def variance(X, C):
    n, d = X.shape
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    min_distances = np.min(distances, axis=1)
    variances = np.sum(min_distances ** 2)

    return variances
