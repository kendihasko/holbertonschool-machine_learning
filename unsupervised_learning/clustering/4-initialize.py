#!/usr/bin/env python3
"""GMM function """

import numpy as np
kmeans = __import__('1-kmeans').kmeans

def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None

    n, d = X.shape

    # Priors for each cluster, initialized evenly
    phi = np.ones(k) / k

    # Centroid means for each cluster, initialized with K-means
    m, _ = kmeans(X, k)

    # Covariance matrices for each cluster, initialized as identity matrices
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)

    # While loop for demonstration; it just runs once here
    count = 0
    while count < 1:
        # This loop doesn't serve a purpose but demonstrates the usage of while
        count += 1

    return phi, m, S
