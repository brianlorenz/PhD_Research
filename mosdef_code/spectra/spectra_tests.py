# Functions for testing how spectra scale

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
from spectra_funcs import clip_skylines, get_spectra_files, median_bin_spec, read_spectrum, get_too_low_gals, norm_spec_sed
import matplotlib.patches as patches


def test_spec(groupID, thresh=0.1):
    """Plots the spectrum

    Parameters:
    groupID (int): id of the group that you are working with
    thresh (float): From 0 to 1, fraction where less than this percentage of the group will be marked as a bad part of the spectrum

    Returns:
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    composite_sed = read_composite_sed(groupID)

    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)

    mosdef_objs = [get_mosdef_obj(field, int(v4id))
                   for field, v4id in fields_ids]

    for mosdef_obj in mosdef_objs[1:3]:
        z_spec = mosdef_obj['Z_MOSFIRE']
        field = mosdef_obj['FIELD_STR']
        v4id = mosdef_obj['V4ID']
        spectra_files = get_spectra_files(mosdef_obj)
        for spectrum_file in spectra_files:
            # Looping over each file, open the spectrum
            spec_loc = imd.spectra_dir + '/' + spectrum_file
            hdu_spec = fits.open(spec_loc)[1]
            hdu_errs = fits.open(spec_loc)[2]
            spec_data = hdu_spec.data
            spec_data_errs = hdu_errs.data

            # Compute the wavelength range
            wavelength = (1. + np.arange(hdu_spec.header["naxis1"]) - hdu_spec.header[
                "crpix1"]) * hdu_spec.header["cdelt1"] + hdu_spec.header["crval1"]

            # Clip the skylines:
            spec_data_clip, mask = clip_skylines(
                wavelength, spec_data, spec_data_errs)

            # De-redshift
            rest_wavelength = wavelength / (1 + z_spec)

            norm_factor, spec_correlate, used_points = norm_spec_sed(
                composite_sed, spec_data_clip, wavelength, mask)
            spec_data_norm = spec_data_clip * norm_factor

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            ax.plot(rest_wavelength, spec_data, color='blue',
                    lw=1, label='Spectrum')
            ax.plot(rest_wavelength, spec_data_norm, color='orange',
                    lw=1, label='Spectrum Normalized')
            ax.plot(composite_sed['rest_wavelength'], composite_sed['f_lambda'], color='black',
                    lw=1, label='composite Sed')

            ax.set_ylim(-1 * 10**-18, 1.01 * np.max(spec_data_norm))
            ax.legend(loc=1, fontsize=axisfont - 3)

            ax.set_xlabel(
                'Wavelength ($\\rm{\AA}$)', fontsize=axisfont)

            ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
            ax.tick_params(labelsize=ticksize, size=ticks)

            plt.show()
            # fig.savefig(imd.cluster_dir +
            # f'/composite_spectra/{norm_method}/{groupID}_spectrum.pdf')
            plt.close()
