#

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from axis_ratio_funcs import read_axis_ratio


def plot_axis_ratios(n_clusters, filt):
    """Create a histogram of similarities between individual galaxies in each cluser, and also to the composite SED

    Parameters:
    n_clusters (int): Number of clusters
    filt(int): Filter to read, either 125, 140, or 160

    Returns:
    """
    save_dir = imd.cluster_dir + '/cluster_stats/axis_ratios/'

    zobjs = ascii.read(
        imd.cluster_dir + '/zobjs_clustered.csv', data_start=1).to_pandas()
    zobjs['new_index'] = zobjs.index

    for groupID in range(n_clusters):
        print(f'Computing Axis Ratios for Cluster {groupID}')
        galaxies = zobjs[zobjs['cluster_num'] == groupID]

        fields_ids = [(galaxies.iloc[i]['field'], galaxies.iloc[i]['v4id'])
                      for i in range(len(galaxies))]

        ar_df = read_axis_ratio(filt, fields_ids)

        axisfont = 14
        ticksize = 12
        ticks = 8
        titlefont = 24
        legendfont = 14
        textfont = 16

        # Figure for just the galaixes in that cluster
        fig, ax = plt.subplots(figsize=(8, 7))

        bins = np.arange(0, 1.05, 0.05)
        ax.hist(ar_df['axis_ratio'], bins=bins, color='black')

        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel('Axis Ratio', fontsize=axisfont)
        ax.set_ylabel('Number of Galaxies', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)
        fig.savefig(save_dir + f'{groupID}_axis_ratio_hist.pdf')
        plt.close()
