import numpy as np
from cluster_algorithms.cross_cor_eqns import get_cross_cor

def L2_norm(X):
    l2_norms = np.sqrt(np.sum(X**2, axis=1))
    for i in range(len(X)):
        X[i] = X[i]/l2_norms[i]
    return X

def cross_cor_norm(X, sed):
    a12_scalings = [get_cross_cor(pixel, sed)[0] for pixel in X]
    for i in range(len(X)):
        X[i] = X[i] - sed*a12_scalings[i] 
    return X 