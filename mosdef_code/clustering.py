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
import shutil


# X = np.array([[1, 1],
#               [2, 1],
#               [1, 0],
#               [4, 7],
#               [3, 5],
#               [3, 6]])

# Need to replace this affinity matrix with the matrix of cross correlation values between the points. Ones along the diagonal, and values between 0 and 1
# aff_matrix = np.array([[1, 1/2, 2/3, 0.3, 1/18, 1/20],
#                        [1/2, 1, 1/3, 1/9, 1/5, 1/7],
#                        [2/3, 1/3, 1, 1/40, 1/26, 1/30],
#                        [0.3, 1/9, 1/40, 1, 1/5, 1/4],
#                        [1/18, 1/5, 1/26, 1/5, 1, 2/3],
#                        [1/20, 1/7, 1/30, 1/4, 2/3, 1]])

# xvals = [X[i][0] for i in range(len(X))]
# yvals = [X[i][1] for i in range(len(X))]

# clusters = clustering.labels_
# plt.scatter(xvals, yvals, c=clusters_aff)

cluster_dir = '/Users/galaxies-air/mosdef/Clustering/'


def cluster_seds(n_clusters):
    """Read in the similarity matrix and cluster the SEDs

    Parameters:
    n_clusters (int) - number of clusters to use

    """

    affinity_matrix = ascii.read(
        cluster_dir+'similarity_matrix.csv').to_pandas().to_numpy()

    zobjs_df = ascii.read(
        cluster_dir+'zobjs_order.csv', data_start=1).to_pandas()
    zobjs_df.columns = ['new_index', 'original_zobjs_index', 'field', 'v4id']

    clustering_aff = SpectralClustering(
        n_clusters=n_clusters, assign_labels="discretize", random_state=0, affinity='precomputed').fit(affinity_matrix)

    clusters = clustering_aff.labels_
    cluster_num_df = pd.DataFrame(clusters, columns=['cluster_num'])
    zobjs_df = zobjs_df.merge(
        cluster_num_df, left_index=True, right_index=True)

    zobjs_df.to_csv(
        '/Users/galaxies-air/mosdef/Clustering/zobjs_clustered.csv', index=False)

    for i in range(n_clusters):
        os.mkdir(cluster_dir+str(i))

    for i in range(len(zobjs_df)):
        obj = zobjs_df.iloc[i]
        filename = f'{obj["field"]}_{obj["v4id"]}_mocktest.pdf'
        print(filename)
        print(cluster_dir+f'{obj["cluster_num"]}/'+filename)
        shutil.copy(f'/Users/galaxies-air/mosdef/SED_Images/mock_sed_images/'+filename, cluster_dir+f'{obj["cluster_num"]}/'+filename)

    return


#eigenvals, eignvectors = np.linalg.eig(affinity_matrix)
