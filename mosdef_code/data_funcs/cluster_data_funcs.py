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


def find_dissimilar_clusters(n_clusters, thresh=0.7, verb=False):
    """Searches the clusters and returns the groupIDs of the ones that have average similarities to composite less than thresh

    Parameters:
    n_clusters (int): Number of clusters
    thresh (int): cutoff threshold, range 0 to 1. Average similarity must be less than this to be flagged
    verb (boolean): set to true to print avg values

    Returns:
    dissimilar_clusters (list): Lists the groupIDs of the clusters that have average similarities less than thresh
    """

    dissimilar_clusters = []
    for groupID in range(n_clusters):
        similarity_composite_location = imd.cluster_dir + f'/cluster_stats/similarities/similarities_composite/{groupID}_similarity_composite.csv'
        similarity_composite_df = ascii.read(
            similarity_composite_location).to_pandas()
        sim_values = similarity_composite_df['similarity_composite']
        avg = np.mean(sim_values)
        if verb:
            print(f'Average similarity for {groupID}: {avg}')
        if avg < thresh:
            dissimilar_clusters.append(groupID)
    return dissimilar_clusters


def find_bad_clusters(n_clusters):
    """Searches the clusters and returns the groupIDs of the ones that are either low or dissimilar

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    bad_clusters (list): Lists the groupIDs of the clusters fail either low or dissimilar tests
    """
    low_clusters = find_low_clusters(n_clusters)
    dissimilar_clusters = find_dissimilar_clusters(n_clusters)
    bad_clusters = list(dict.fromkeys(low_clusters + dissimilar_clusters))
    return np.sort(bad_clusters)


def get_cluster_fields_ids(groupID):
    """Gets the list of all files and the fields and ids for each galaxy in a given cluster

    Parameters:
    groupID (int): GroupID of the cluster to perform this on

    Returns:
    cluster_names (list): list of the files in that cluster
    fields_ids (list of tuples): Each tuple is of the form (field, v4id) for each object in the cluster
    """
    # Read in the galaxies from that cluster
    cluster_names = os.listdir(imd.cluster_dir + '/' + str(groupID))
    # Splits into list of tuples: [(field, v4id), (field, v4id), (field,
    # v4id), ...]
    fields_ids = [(line.split('_')[0], line.split('_')[1])
                  for line in cluster_names]
    return cluster_names, fields_ids
