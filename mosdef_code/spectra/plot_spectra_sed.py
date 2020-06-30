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
from spectra_funcs import clip_skylines, get_spectra_files, median_bin_spec
import matplotlib.patches as patches


def plot_spectrum_sed(groupID, binsize):
    """Overplots a smoothed spectrum on the SED

    Parameters:
    groupID (int): ID of the cluster to perform the stacking
    binsize (int): how many points to put in each bin when smoothing the spectrum

    Returns:
    """

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    total_spec_df = ascii.read(
        imd.cluster_dir + f'/composite_spectra/{groupID}_spectrum.csv').to_pandas()
    composite_sed = read_composite_sed(groupID)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    spectrum = total_spec_df['f_lambda']
    wavelength = total_spec_df['wavelength']

    wave_bin, spec_bin = median_bin_spec(wavelength, spectrum, binsize=binsize)

    # ax.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
    ax.plot(wave_bin, spec_bin, color='blue',
            lw=1, label='Median Binned Spectrum')
    ax.errorbar(composite_sed['rest_wavelength'], composite_sed['f_lambda'],
                yerr=[composite_sed['err_f_lambda_d'], composite_sed['err_f_lambda_u']], ls='', marker='o', markersize=4, color='black')

    if np.min(spec_bin) < 0:
        min_y = 1.01 * np.min([np.min(spec_bin),
                               np.min(composite_sed['f_lambda'])])
    else:
        min_y = 0.99 * np.min([np.min(spec_bin),
                               np.min(composite_sed['f_lambda'])])
    ax.set_ylim(min_y, 1.01 * np.max([np.max(spec_bin),
                                      np.max(composite_sed['f_lambda'])]))
    ax.legend(loc=2, fontsize=axisfont - 3)

    ax.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    fig.savefig(imd.cluster_dir + f'/composite_spectra/{groupID}_spectrum_sed.pdf')
    plt.close()


def plot_all_spectrum_sed(n_clusters, binsize):
    """Overplots a smoothed spectrum on the SED for all clusters

    Parameters:
    n_clusters (int): Number of clusters of galaxies
    binsize (int): how many points to put in each bin when smoothing the spectrum

    Returns:
    """
    for groupID in range(n_clusters):
        plot_spectrum_sed(groupID, binsize)
