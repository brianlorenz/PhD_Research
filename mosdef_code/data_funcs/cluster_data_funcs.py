# Functions that deal with the clusters

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
from read_data import mosdef_df
import initialize_mosdef_dirs as imd


def find_low_clusters(n_clusters, thresh=5):
    """Searches the clusters and returns the groupIDs of the ones that have fewer galaxies than thresh

    Parameters:
    n_clusters (int): Number of clusters
    thresh (int): cutoff threshold. Any cluster with fewer than this many galaxies is identified

    Returns:
    low_clusters (list): Lists the groupIDs of the clusters that have fewer than thresh galaxies
    """

    low_clusters = []
    for groupID in range(n_clusters):
        gals = os.listdir(imd.cluster_dir + '/' + str(groupID))
        n_gals = len(gals)
        if n_gals < thresh:
            low_clusters.append(groupID)
    return low_clusters
