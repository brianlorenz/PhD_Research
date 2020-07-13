# Codes for Plotting the composite spectra over the composite SEDs
# plot_spectrum_sed(groupID, binsize)
# plot_all_spectrum_sed(n_clusters, binsize)

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
from spectra_funcs import clip_skylines, smooth_spectrum, get_spectra_files, median_bin_spec, norm_spec_sed, get_too_low_gals, read_spectrum, read_composite_spectrum
import matplotlib.patches as patches


def plot_spectrum_sed(groupID, norm_method, binsize):
    """Overplots a smoothed spectrum on the SED

    Parameters:
    groupID (int): ID of the cluster to perform the stacking
    norm_method (str): Method used to normalize the galaxies - e.g. 'composite_filter', 'cluster_norm'
    binsize (int): how many points to put in each bin when smoothing the spectrum

    Returns:
    """

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    total_spec_df = read_composite_spectrum(groupID, norm_method)
    composite_sed = read_composite_sed(groupID)

    # norm, b12, used_values_df = norm_spec_sed(composite_sed, total_spec_df)
    # print(f'Normalized by {norm}')
    # print(f'Correlation factor: {1 - b12}')
    norm = 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    spectrum = total_spec_df['f_lambda']
    wavelength = total_spec_df['wavelength']

    # wave_bin, spec_bin = median_bin_spec(wavelength, spectrum,
    # binsize=binsize)
    smooth_spec = smooth_spectrum(spectrum, width=binsize)

    too_low_gals, plot_cut, not_plot_cut, n_gals_in_group, cutoff, cut_wave_high, cut_wave_low = get_too_low_gals(
        groupID, norm_method)

    # ax.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
    # ax.plot(wave_bin, spec_bin * norm, color='blue',
    #         lw=1, label='Median Binned Spectrum')
    ax.plot(wavelength, smooth_spec, color='blue',
            lw=1, label='Smoothed Spectrum')
    ax.plot(wavelength[wavelength < cut_wave_low], smooth_spec[wavelength < cut_wave_low], color='red',
            lw=1)
    ax.plot(wavelength[wavelength > cut_wave_high], smooth_spec[wavelength > cut_wave_high], color='red',
            lw=1)
    # ax.plot(wave_bin[wave_bin < cut_wave_low], spec_bin[wave_bin < cut_wave_low] * norm, color='red',
    #         lw=1)
    # ax.plot(wave_bin[wave_bin > cut_wave_high], spec_bin[wave_bin > cut_wave_high] * norm, color='red',
    #         lw=1)
    # ax.plot(wave_bin[plot_cut], spec_bin[plot_cut] * norm, color='red',
    #         lw=1, label=f'Too Low ({cutoff})')
    # ax.plot(wave_bin[not_plot_cut], spec_bin[not_plot_cut] * norm, color='red',
    #         lw=1)
    # ax.plot(used_values_df['wavelength'],
    #         used_values_df['sed_flux'], color='orange', ls='None', marker='o')
    # ax.plot(used_values_df['wavelength'],
    # used_values_df['spectrum_flux'] * norm, color='purple', ls='None',
    # marker='o')
    ax.errorbar(composite_sed['rest_wavelength'], composite_sed['f_lambda'],
                yerr=[composite_sed['err_f_lambda_d'], composite_sed['err_f_lambda_u']], ls='', marker='o', markersize=4, color='black')

    # if np.min(spec_bin * norm) < 0:
    #     min_y = 1.01 * np.min([np.min(spec_bin * norm),
    #                            np.min(composite_sed['f_lambda'])])
    # else:
    min_y = 0.99 * np.min([np.min(smooth_spec[smooth_spec > 0] * norm),
                           np.min(composite_sed['f_lambda'])])
    ax.set_ylim(min_y, 1.01 * np.max(composite_sed['f_lambda']))
    ax.legend(loc=1, fontsize=axisfont - 3)

    ax.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    fig.savefig(imd.cluster_dir + f'/composite_spectra/{norm_method}/{groupID}_spectrum_sed.pdf')
    plt.close()


def plot_all_spectrum_sed(n_clusters, norm_method, binsize):
    """Overplots a smoothed spectrum on the SED for all clusters

    Parameters:
    n_clusters (int): Number of clusters of galaxies
    norm_method (str): Method used to normalize the galaxies - e.g. 'composite_filter', 'cluster_norm'
    binsize (int): how many points to put in each bin when smoothing the spectrum

    Returns:
    """
    for groupID in range(n_clusters):
        plot_spectrum_sed(groupID, norm_method, binsize)
