import numpy as np
from itertools import combinations_with_replacement
from cluster_algorithms.cross_cor_eqns import get_cross_cor
from sklearn.cluster import SpectralClustering
import time
import matplotlib.pyplot as plt

def spectral_cluster_cross_cor(pixel_seds, *args):
    X = pixel_seds.T
    n_pixels = len(X)

    sim_matrix = np.zeros((n_pixels, n_pixels))

    t0 = time.time()
    for i, j in combinations_with_replacement(range(n_pixels), 2):
        _, b12 = get_cross_cor(X[i], X[j])
        sim_matrix[i, j] = 1-b12 # Need to do 1-b12 since we want identical items to have a score of 1
        sim_matrix[j, i] = 1-b12
    t1 = time.time()
    print(f'Computed similarity matrix in {t1-t0} seconds')

    eigenvals, eignvectors = np.linalg.eig(sim_matrix)
    x_axis = np.arange(1, len(eigenvals)+1, 1)
    dx = 1
    derivative = np.gradient(eigenvals, dx)
    plt.plot(x_axis, eigenvals, ls='-', marker='o', color='black')
    # plt.plot(x_axis, derivative, ls='-', marker='o', color='orange')
    plt.xlim(0, 15)
    # plt.show()

    clustering_aff = SpectralClustering(n_clusters=4, assign_labels="discretize", random_state=0, affinity='precomputed').fit(sim_matrix)

    cluster_values = clustering_aff.labels_ + 1
    
    return cluster_values

