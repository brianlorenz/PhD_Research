from sklearn.cluster import AgglomerativeClustering
import numpy as np
from cluster_algorithms.normalize import normalize_X
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def agglomerative_cluster(pixel_seds, sed, *args, norm_method=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)

    agg = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')


    clustering_labels = agg.fit_predict(X)


    cluster_values = clustering_labels + 1

    
    return cluster_values