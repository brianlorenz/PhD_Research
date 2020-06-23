# Contains functions that perform cross correlation between the mock seds

import sys
import os
import string
import pdb
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
from read_data import mosdef_df
from mosdef_obj_data_funcs import get_mosdef_obj, read_sed
from plot_funcs import populate_main_axis
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd


def get_cross_cor(mock_sed_1, mock_sed_2):
    """Create a mock SED at standard wavelengths

    Parameters:
    mock_sed_1 (pd.DataFrame): read the SED, then put it into a dataframe and directly into this funciton
    mock_sed_2 (pd.DataFrame): read the SED, then put it into a dataframe and directly into this funciton

    Returns:
    """

    # WANT TO RUN THIS 100x LIKE THE OTHER TO GET UNCERTAINTY?
    f1 = mock_sed_1['f_lambda']
    f2 = mock_sed_2['f_lambda']
    a12 = np.sum(f1 * f2) / np.sum(f2**2)
    b12 = np.sqrt(np.sum((f1 - a12 * f2)**2) / np.sum(f1**2))
    return b12


def read_mock_sed(field, v4id):
    """Reads one of the mock seds


    """
    sed_loc = imd.home_dir + f'/mosdef/mock_sed_csvs/{field}_{v4id}_sed.csv'
    sed = ascii.read(sed_loc).to_pandas()
    return sed


def correlate_all_seds(zobjs):
    """Creates the similarity matrix between all SEDs


    Returns:
    similarity_matrix
    zobjs_df - zobjs dataframe (with indices) that
    """
    fields = [zobjs[i][0] for i in range(len(zobjs))]
    v4ids = [zobjs[i][1] for i in range(len(zobjs))]
    zobjs_df = pd.DataFrame(zip(fields, v4ids), columns=['field', 'v4id'])
    zobjs_df['v4id'] = zobjs_df['v4id'].astype(int)
    badidx = zobjs_df[zobjs_df.v4id < 0].index
    zobjs_df = zobjs_df.drop(badidx)

    # INDICES RESET
    zobjs_df = zobjs_df.reset_index()
    dimension = len(zobjs_df)
    similarity_matrix = np.zeros(shape=(dimension, dimension))

    for i in range(dimension):
        sed_i = read_mock_sed(
            zobjs_df.iloc[i]['field'], zobjs_df.iloc[i]['v4id'])
        for j in range(dimension - i):
            sed_j = read_mock_sed(
                zobjs_df.iloc[i + j]['field'], zobjs_df.iloc[i + j]['v4id'])
            similarity_matrix[i, i + j] = 1 - get_cross_cor(sed_i, sed_j)
            similarity_matrix[i + j, i] = similarity_matrix[i, i + j]
    zobjs_df.to_csv(imd.home_dir + '/mosdef/Clustering/zobjs_order.csv')
    np.savetxt(imd.home_dir + '/mosdef/Clustering/similarity_matrix.csv',
               similarity_matrix, delimiter=',')
    return None
