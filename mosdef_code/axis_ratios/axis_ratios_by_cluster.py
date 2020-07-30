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
from emission_measurements import read_emission_df, get_emission_measurements


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


def plot_axis_ratios_balmer_dec(n_clusters, filt):
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
        emission_df = read_emission_df()

        balmer_decs = []
        balmer_errs = []
        for obj in fields_ids:
            row = get_emission_measurements(emission_df, obj)
            # Object without measurement in a line
            # Drop objects without HA or HB measurement
            row = row[row['HA6565_FLUX'] > 0]
            row = row[row['HB4863_FLUX'] > 0]
            if len(row) < 1:
                balmer_decs.append(-99)
                balmer_errs.append(-99)
                print('No measurement found for one or both lines')
                continue
            if len(row) > 1:
                decs = []
                errs = []
                for i in range(len(row)):
                    single_row = row.iloc[i]
                    balmer_dec, balmer_dec_err = compute_balmer_dec(single_row)
                    # Skip it if the error is too large AM I ALLOWED TO DO THIS
                    if balmer_dec_err > 5:
                        continue
                    decs.append(balmer_dec)
                    errs.append(balmer_dec_err)
                average_dec = np.mean(decs)
                variances = [i**2 for i in errs]
                summed_variances = np.sum(variances)
                average_err = summed_variances / len(variances)
                print(f'Measured Balmer dec from multiple spectra as {average_dec}')
                balmer_decs.append(average_dec)
                balmer_errs.append(average_err)
            else:
                row = row.iloc[0]
                balmer_dec, balmer_dec_err = compute_balmer_dec(row)
                print(f'Measured Balmer dec from one spectrum as {balmer_dec}')
                balmer_decs.append(balmer_dec)
                balmer_errs.append(balmer_dec_err)

        balmer_df = pd.DataFrame(zip(balmer_decs, balmer_errs), columns=[
                                 'balmer_dec', 'balmer_dec_err'])

        balmer_df.to_csv(save_dir + f'{groupID}_F{filt}_balmer_decs.csv', index=False)

        axisfont = 14
        ticksize = 12
        ticks = 8
        titlefont = 24
        legendfont = 14
        textfont = 16

        # Figure for just the galaixes in that cluster
        fig, ax = plt.subplots(figsize=(8, 7))

        # Filter out those with -99
        filt_out = balmer_df['balmer_dec'] > -10
        ar_df = ar_df.reset_index()

        bins = np.arange(0, 1.05, 0.05)
        ax.errorbar(ar_df[filt_out]['axis_ratio'], balmer_df[filt_out]['balmer_dec'], xerr=ar_df[filt_out][
                    'err_axis_ratio'], yerr=balmer_df[filt_out]['balmer_dec_err'], color='black', ls='None', marker='o')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-2, 10)
        ax.set_xlabel('Axis Ratio', fontsize=axisfont)
        ax.set_ylabel('Balmer Decrement', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)
        fig.savefig(save_dir + f'{groupID}_F{filt}_axis_ratio_balmer.pdf')
        plt.close()


def plot_all_balmer_decs(n_clusters, filt):
    """Create a histogram of similarities between individual galaxies in each cluser, and also to the composite SED

    Parameters:
    n_clusters (int): Number of clusters
    filt (int): Filter to read, either 125, 140, or 160

    Returns:
    """

    save_dir = imd.cluster_dir + '/cluster_stats/axis_ratios/'

    zobjs = ascii.read(
        imd.cluster_dir + '/zobjs_clustered.csv', data_start=1).to_pandas()
    zobjs['new_index'] = zobjs.index

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    bins = np.arange(0, 1.05, 0.05)
    for groupID in range(n_clusters):
        galaxies = zobjs[zobjs['cluster_num'] == groupID]
        fields_ids = [(galaxies.iloc[i]['field'], galaxies.iloc[i]['v4id'])
                      for i in range(len(galaxies))]
        ar_df = read_axis_ratio(filt, fields_ids)
        balmer_df = ascii.read(save_dir + f'{groupID}_F{filt}_balmer_decs.csv').to_pandas()
        # Filter out those with -99
        filt_out = balmer_df['balmer_dec'] > -10
        ar_df = ar_df.reset_index()
        ax.errorbar(ar_df[filt_out]['axis_ratio'], balmer_df[filt_out]['balmer_dec'], xerr=ar_df[filt_out][
            'err_axis_ratio'], yerr=balmer_df[filt_out]['balmer_dec_err'], color='black', ls='None', marker='o')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-2, 10)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('Balmer Decrement', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + f'All_F{filt}_axis_ratio_balmer.pdf')
    plt.close()


def compute_balmer_dec(row):
    """Calculate the Balmer decrement and error, given a row in the emission_df

    Parameters:
    row (pd.DataFrame): A row from the emission_df with nonzero HA and HB

    Returns:
    """
    balmer_dec = row['HA6565_FLUX'] / row['HB4863_FLUX']
    balmer_dec_err = balmer_dec * \
        ((row['HA6565_FLUX_ERR'] / row['HA6565_FLUX']) +
         (row['HB4863_FLUX_ERR'] / row['HB4863_FLUX']))
    return balmer_dec, balmer_dec_err
