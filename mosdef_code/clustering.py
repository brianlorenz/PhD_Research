# Finds clusters out of an affinity matrix

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import get_mosdef_obj, read_sed
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

X = np.array([[1, 1],
              [2, 1],
              [1, 0],
              [4, 7],
              [3, 5],
              [3, 6]])


# Need to replace this affinity matrix with the matrix of cross correlation values between the points. Ones along the diagonal, and values between 0 and 1
aff_matrix = np.array([[1, 1/2, 2/3, 0.3, 1/18, 1/20],
                       [1/2, 1, 1/3, 1/9, 1/5, 1/7],
                       [2/3, 1/3, 1, 1/40, 1/26, 1/30],
                       [0.3, 1/9, 1/40, 1, 1/5, 1/4],
                       [1/18, 1/5, 1/26, 1/5, 1, 2/3],
                       [1/20, 1/7, 1/30, 1/4, 2/3, 1]])

xvals = [X[i][0] for i in range(len(X))]
yvals = [X[i][1] for i in range(len(X))]

clustering = SpectralClustering(
    n_clusters=4, assign_labels="discretize", random_state=0).fit(X)
clustering_aff = SpectralClustering(
    n_clusters=3, assign_labels="discretize", random_state=0, affinity='precomputed').fit(aff_matrix)

clusters = clustering.labels_
clusters_aff = clustering_aff.labels_

plt.scatter(xvals, yvals, c=clusters_aff)
plt.show()
