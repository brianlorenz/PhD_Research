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


def plot_axis_ratio_clusters_balmer_dec(n_groups):
    """

    Parameters:
    n_groups (int): NUmber of axis ratio groups
    """
    save_dir = imd.cluster_dir + '/axis_ratio_cluster_plots/'

    fit_dfs = [ascii.read(imd.cluster_dir + f'/emission_fitting/axis_ratio_clusters/{axis_group}_emission_fits.csv').to_pandas() for axis_group in range(n_groups)]
    ar_dfs = [ascii.read(imd.cluster_dir + f'/composite_spectra/axis_stack/{axis_group}_df.csv').to_pandas() for axis_group in range(n_groups)]

    medians = [np.median(ar_df['use_ratio']) for ar_df in ar_dfs]
    stds = [np.std(ar_df['use_ratio']) for ar_df in ar_dfs]

    balmer_decs = []
    balmer_errs = []
    for i in range(len(fit_dfs)):
        fit_df = fit_dfs[i]
        ha_flux = float(fit_df[fit_df['line_name'] == 'Halpha']['flux'])
        err_ha_flux = float(
            fit_df[fit_df['line_name'] == 'Halpha']['err_flux'])
        hb_flux = float(fit_df[fit_df['line_name'] == 'Hbeta']['flux'])
        err_hb_flux = float(fit_df[fit_df['line_name'] == 'Hbeta']['err_flux'])
        balmer_dec = ha_flux / hb_flux
        balmer_decs.append(balmer_dec)
        balmer_dec_err = balmer_dec * \
            ((err_ha_flux / ha_flux) +
             (err_hb_flux / hb_flux))
        breakpoint()
        balmer_errs.append(balmer_dec_err)
    breakpoint()

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Figure for just the galaixes in that cluster
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.errorbar(medians, balmer_decs, xerr=stds, yerr=balmer_errs,
                color='black', ls='None', marker='o')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Axis Ratio', fontsize=axisfont)
    ax.set_ylabel('Balmer Decrement', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(save_dir + f'AR_Clusters_Balmer_Decs.pdf')
    plt.close()
