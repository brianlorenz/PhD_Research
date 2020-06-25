# Codes for stacking MOSDEF spectra within clusters

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
from spectra_funcs import clip_skylines, get_spectra_files


def stack_spectra(groupID):
    """Stack all the spectra for every object in a given group

    Parameters:
    groupID (int): ID of the cluster to perform the stacking

    Returns:
    """

    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)

    mosdef_objs = [get_mosdef_obj(field, int(v4id))
                   for field, v4id in fields_ids]

    #min, max, step-size
    spectrum_wavelength = np.arange(3000, 10000, 1)

    # Now that we have the mosdef objs for each galaxy in the cluster, we need
    # to loop over each one
    cluster_spectra_dfs = []
    interp_cluster_spectra_dfs = []
    for mosdef_obj in mosdef_objs:
        # Get the redshift and normalization
        z_spec = mosdef_obj['Z_MOSFIRE']
        field = mosdef_obj['FIELD_STR']
        v4id = mosdef_obj['V4ID']
        norm_sed = read_sed(field, v4id, norm=True)
        norm_factor = np.median(norm_sed['norm_factor'])
        print(f'Reading Spectra for {field} {v4id}, z={z_spec:.3f}')
        # Find all the spectra files corresponding to this object
        spectra_files = get_spectra_files(mosdef_obj)
        for spectrum_file in spectra_files:
            # Looping over each file, open the spectrum
            spec_loc = imd.spectra_dir + spectrum_file
            hdu = fits.open(spec_loc)[1]
            spec_data = hdu.data

            if 'H' in spectrum_file:
                filt = 'H'
            if 'K' in spectrum_file:
                filt = 'K'
            if 'J' in spectrum_file:
                filt = 'J'
            if 'Y' in spectrum_file:
                filt = 'Y'

            # Compute the wavelength range
            wavelength = (1. + np.arange(hdu.header["naxis1"]) - hdu.header[
                "crpix1"]) * hdu.header["cdelt1"] + hdu.header["crval1"]

            # Clip the skylines:
            spec_data = clip_skylines(wavelength, spec_data, filt)

            # De-redshift
            rest_wavelength = wavelength / (1 + z_spec)

            # NORMALZE - HOW BEST TO DO THIS?
            spec_data_norm = spec_data * norm_factor

            norm_interp = interpolate.interp1d(
                rest_wavelength, spec_data_norm, fill_value=0, bounds_error=False)

            spectrum_flux_norm = norm_interp(spectrum_wavelength)

            # Save one instance that contians just the data
            spectrum_df = pd.DataFrame(zip(rest_wavelength, spec_data, spec_data_norm),
                                       columns=['rest_wavelength', 'f_lambda', 'f_lambda_norm'])

            # Save a second instance that conatins the interpolated spectrum
            interp_spectrum_df = pd.DataFrame(zip(spectrum_wavelength, spectrum_flux_norm),
                                              columns=['rest_wavelength', 'f_lambda_norm'])
            cluster_spectra_dfs.append(spectrum_df)
            interp_cluster_spectra_dfs.append(interp_spectrum_df)

    # Pulls out just the flux values of each spectrum
    norm_interp_specs = [interp_cluster_spectra_dfs[i]['f_lambda_norm']
                         for i in range(len(interp_cluster_spectra_dfs))]
    # For each wavelength, counts the number of spectra that are nonzero
    number_specs_by_wave = np.count_nonzero(norm_interp_specs, axis=0)

    summed_spec = np.sum(norm_interp_specs, axis=0)
    #total_spec = (summed_spec / number_specs_by_wave)
    # Have to use divz since lots of zeros
    total_spec = divz(summed_spec, number_specs_by_wave)
    total_spec_df = pd.DataFrame(zip(spectrum_wavelength, total_spec),
                                 columns=['wavelength', 'f_lambda'])

    total_spec_df.to_csv(
        imd.cluster_dir + f'/composite_spectra/{groupID}_spectrum.csv', index=False)

    plot_spec(groupID)
    return


def plot_spec(groupID):
    """Plots the spectrum

    Parameters:
    total_spec_df (pd.DataFrame): Dataframe from above containing wavelength vs f_lambda, all normalizeda and stacked
    groupID (int): id of the group that you are working with

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

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    spectrum = total_spec_df['f_lambda']
    wavelength = total_spec_df['wavelength']

    ax.plot(wavelength, spectrum, color='black', lw=1)

    ax.set_ylim(-1 * 10**-18, 1 * 10**-18)
    # ax.set_xlim()

    plt.tight_layout()
    fig.savefig(imd.cluster_dir + f'/composite_spectra/{groupID}_spectrum.pdf')
    plt.close()


def divz(X, Y):
    return X / np.where(Y, Y, Y + 1) * np.not_equal(Y, 0)
