# Plots some statistics about each cluser
# plot_similarity can plot the similarities between each pair of galaxies
# in a cluster and each galaxy and its SED

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, read_mock_composite_sed
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from cross_correlate import get_cross_cor


def plot_similarity(n_clusters):
    """Create a histogram of similarities between individual galaxies in each cluser, and also to the composite SED

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    save_dir = imd.cluster_dir + '/cluster_stats/similarities/'

    similarity_matrix = ascii.read(
        imd.cluster_dir + 'similarity_matrix.csv').to_pandas().to_numpy()
    zobjs = ascii.read(
        imd.cluster_dir + 'zobjs_clustered.csv', data_start=1).to_pandas()
    zobjs['new_index'] = zobjs.index

    for groupID in range(n_clusters):
        print(f'Computing Similarity for Cluster {groupID}')
        galaxies = zobjs[zobjs['cluster_num'] == groupID]
        similarities = []
        similarities_composite = []
        num_galaxies = len(galaxies)
        for i in range(num_galaxies):
            for j in range(num_galaxies - i):
                if i != j:
                    similarities.append(
                        similarity_matrix[galaxies.iloc[i]['new_index'], galaxies.iloc[j]['new_index']])
        mock_composite_sed = read_mock_composite_sed(groupID)
        for i in range(num_galaxies):
            mock_sed = read_mock_sed(
                galaxies.iloc[i]['field'], galaxies.iloc[i]['v4id'])
            similarities_composite.append(
                1 - get_cross_cor(mock_composite_sed, mock_sed)[1])

        #galaxies['similarity_composite'] = similarities_composite
        zobjs.loc[zobjs['cluster_num'] == groupID,
                  'similarity_composite'] = similarities_composite

        galaxies = zobjs.loc[zobjs['cluster_num'] == groupID]

        axisfont = 14
        ticksize = 12
        ticks = 8
        titlefont = 24
        legendfont = 14
        textfont = 16

        # Figure for just the galaixes in that cluster
        fig, ax = plt.subplots(figsize=(8, 7))

        bins = np.arange(0, 1.05, 0.05)
        ax.hist(similarities, bins=bins, color='black')

        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel('Similarity', fontsize=axisfont)
        ax.set_ylabel('Number of pairs', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)
        fig.savefig(save_dir + f'{groupID}_similarity.pdf')
        plt.close()

        # Figure for the correlation with the composite:
        fig, ax = plt.subplots(figsize=(8, 7))

        bins = np.arange(0, 1.05, 0.05)
        ax.hist(similarities_composite, bins=bins, color='black')

        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel('Similarity to Composite', fontsize=axisfont)
        ax.set_ylabel('Number of galaxies', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)
        fig.savefig(save_dir + f'similarities_composite/{groupID}_similarity_composite.pdf')
        plt.close()

        # Also, save the values between each galaxy and the composite
        galaxies.to_csv(save_dir + f'similarities_composite/{groupID}_similarity_composite.csv', index=False)
