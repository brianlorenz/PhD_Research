from sklearn.cluster import AgglomerativeClustering
import numpy as np
from cluster_algorithms.normalize import L2_norm, cross_cor_norm
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def agglomerative_cluster(pixel_seds, sed, *args, norm_method=''):
    X = pixel_seds.T

    # Normalize by L2 norm
    if norm_method=='_L2':
        X = L2_norm(X)

    if norm_method=='_sed':
        # Subtract a median sed from each pixel, normalized to the level of that pixel
        X = cross_cor_norm(X, sed)

    agg = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')


    clustering_labels = agg.fit_predict(X)


    cluster_values = clustering_labels + 1

    
    return cluster_values