#

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed, setup_get_AV, get_AV
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from axis_ratio_funcs import read_axis_ratio, read_interp_axis_ratio
from emission_measurements import read_emission_df, get_emission_measurements
from astropy.table import Table


save_dir = imd.cluster_dir + '/axis_ratio_cluster_plots/'


def plot_axis_ratio_clusters_balmer_dec(n_groups, mass_bins=[]):
    """Plots the balmer dec for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    mass_bins (list): Set to the boundaries to cut at. i.e. [8.6, 9.3] will make 3 bins: <8.6, 8.6-9.3, >9.3
    """
    fit_dfs = [ascii.read(imd.cluster_dir + f'/emission_fitting/axis_ratio_clusters/{axis_group}_emission_fits.csv').to_pandas() for axis_group in range(n_groups)]
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    # MASS DOESN"T WORK - NEED TO RESTACK AND REFIT BY MASS

    def calc_values(ar_dfs, fit_dfs):
        """Calculates the values needed for these plots

        Parameters:
        ar_df
        """

        medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
        stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]

        balmer_decs = []
        balmer_errs = []
        for i in range(len(fit_dfs)):
            fit_df = fit_dfs[i]
            balmer_decs.append(fit_df['balmer_dec'].iloc[0])
            balmer_errs.append(
                (fit_df['err_balmer_dec_low'].iloc[0], fit_df['err_balmer_dec_high'].iloc[0]))
        balmer_errs = np.transpose(np.array(balmer_errs))
        return medians, stds, balmer_decs, balmer_errs

    # This is the variable that we store the calculated info into
    values_sets = []
    if len(mass_bins) > 0:
        ar_df_sets = []
        # Add the masses if we are using them
        for ar_df in ar_dfs:
            l_masses = [get_mosdef_obj(ar_df.iloc[i]['field'], ar_df.iloc[i]['v4id'])[
                'LMASS'] for i in range(len(ar_df))]
            ar_df['LMASS'] = l_masses
        # Start looping over the bins to divide
        for i in range(len(mass_bins)):
            # Special case for the first bin
            if i == 0:
                ar_df_sets.append([ar_df[ar_df['LMASS'] < mass_bins[0]]
                                   for ar_df in ar_dfs])
                continue
            ar_df_sets.append([ar_df[np.logical_and(ar_df['LMASS'] > mass_bins[i - 1], ar_df['LMASS'] < mass_bins[i])]
                               for ar_df in ar_dfs])
        # After making all of the intermediates, make the last bin
        ar_df_sets.append([ar_df[ar_df['LMASS'] > mass_bins[
                          len(mass_bins) - 1]] for ar_df in ar_dfs])
        for ar_df_set in ar_df_sets:
            values_sets.append(calc_values(ar_df_set, fit_dfs))

    else:
        values_sets.append(calc_values(ar_dfs, fit_dfs))

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    def plot_values(value_set, color, label):
        medians = value_set[0]
        stds = value_set[1]
        balmer_decs = value_set[2]
        balmer_errs = value_set[3]
        ax.errorbar(medians, balmer_decs, xerr=stds, yerr=balmer_errs,
                    color='black', ls='None', marker='o')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 8)
    colors = ['black', 'blue', 'orange']
    breakpoint()
    for i in range(len(values_sets)):
        color = colors[i]
        if i == 0:
            try:
                label = f'M<{mass_bins[0]}'
            except:
                label = None
        else:
            try:
                label = f'{mass_bins[i-1]}<M<{mass_bins[i]}'
            except IndexError:
                label = f'M>{mass_bins[-1]}'
        plot_values(values_sets[i], color, label)
    if len(values_sets) > 1:
        ax.legend(fontsize=axisfont)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('Balmer Decrement', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    if mass_bins:
        fig.savefig(save_dir + f'AR_Clusters_Balmer_Decs_{len(mass_bins)+1}_bins.pdf')
    else:
        fig.savefig(save_dir + f'AR_Clusters_Balmer_Decs.pdf')
    plt.close()


def plot_axis_ratio_clusters_mass(n_groups):
    """Plots the mass distribution for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    """
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    # Append each galaxy's mass to its ar info
    for ar_df in ar_dfs:
        l_masses = [get_mosdef_obj(ar_df.iloc[i]['field'], ar_df.iloc[i]['v4id'])[
            'LMASS'] for i in range(len(ar_df))]
        ar_df['LMASS'] = l_masses

    medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
    stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]
    median_mass = [np.median(ar_df['LMASS']) for ar_df in ar_dfs]
    stds_mass = [np.std(ar_df['LMASS']) for ar_df in ar_dfs]

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    for ar_df in ar_dfs:
        [ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['LMASS'], color='gray',
                 marker='.', markersize=6, zorder=1) for i in range(len(ar_df))]

    ax.errorbar(medians, median_mass, xerr=stds, yerr=stds_mass,
                color='black', ls='None', marker='o', zorder=2)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(8, 12)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('log(Stellar Mass)', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + f'AR_Clusters_LMASS.pdf')
    plt.close()


def plot_axis_ratio_clusters_AV(n_groups):
    """Plots the Av distribution for each cluster

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    """
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    fields, av_dfs = setup_get_AV()

    # Append each galaxy's mass to its ar info
    for ar_df in ar_dfs:
        mosdef_objs = [get_mosdef_obj(ar_df.iloc[i]['field'], ar_df.iloc[i][
                                      'v4id']) for i in range(len(ar_df))]
        Avs = [get_AV(fields, av_dfs, mosdef_obj)
               for mosdef_obj in mosdef_objs]
        ar_df['Av'] = Avs

    medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
    stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]
    median_Av = [np.median(ar_df['Av']) for ar_df in ar_dfs]
    stds_Av = [np.std(ar_df['Av']) for ar_df in ar_dfs]

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    for ar_df in ar_dfs:
        [ax.plot(ar_df.iloc[i]['use_ratio'], ar_df.iloc[i]['Av'], color='gray',
                 marker='.', markersize=6, zorder=1) for i in range(len(ar_df))]

    ax.errorbar(medians, median_Av, xerr=stds, yerr=stds_Av,
                color='black', ls='None', marker='o', zorder=2)

    ax.set_xlim(-0.05, 1.05)
    #ax.set_ylim(8, 12)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('FAST A_V', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + f'AR_Clusters_FAST_AV.pdf')
    plt.close()


def plot_re_axis_ratio():
    """Plot of axis ratio vs half-light radius

    Parameters:
    """

    ar_df = read_interp_axis_ratio()
    filter_bad_ratios = ar_df['err_use_ratio'] > 0.1

    axis_ratios = ar_df['use_ratio']
    err_axis_ratios = ar_df['err_use_ratio']

    ar_df_big = ascii.read(
        imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_F125W_galfit_v4.0.csv')
    half_light = ar_df_big['re']
    err_half_light = ar_df_big['dre']

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.errorbar(axis_ratios, half_light, xerr=err_axis_ratios, yerr=err_half_light,
                color='black', ls='None', marker='o', markersize=4)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 2)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('Half-Light Radius (unit)', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + f'AxisRatio_HalfLight.pdf')
    plt.close()
