from sklearn.cluster import DBSCAN
import numpy as np
from cluster_algorithms.normalize import L2_norm, cross_cor_norm


def dbscan_clustering(pixel_seds, sed, *args, norm_method=''):
    X = pixel_seds.T

    # Normalize by L2 norm
    if norm_method=='_L2':
        X = L2_norm(X)

    if norm_method=='_sed':
        # Subtract a median sed from each pixel, normalized to the level of that pixel
        X = cross_cor_norm(X, sed)


    clustering = DBSCAN(eps=0.05, min_samples=10).fit(X)

    cluster_values = clustering.labels_ + 1

    
    return cluster_values