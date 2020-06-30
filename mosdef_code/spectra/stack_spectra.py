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
from spectra_funcs import clip_skylines, get_spectra_files, median_bin_spec
import matplotlib.patches as patches


def stack_spectra(groupID):
    """Stack all the spectra for every object in a given group

    Parameters:
    groupID (int): ID of the cluster to perform the stacking

    Returns:
    """

    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)

    mosdef_objs = [get_mosdef_obj(field, int(v4id))
                   for field, v4id in fields_ids]

    # min, max, step-size
    spectrum_wavelength = np.arange(3000, 10000, 1)

    # Now that we have the mosdef objs for each galaxy in the cluster, we need
    # to loop over each one
    cluster_spectra_dfs = []
    interp_cluster_spectra_dfs = []
    norm_factors = []
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
            hdu_spec = fits.open(spec_loc)[1]
            hdu_errs = fits.open(spec_loc)[2]
            spec_data = hdu_spec.data
            spec_data_errs = hdu_errs.data

            # Compute the wavelength range
            wavelength = (1. + np.arange(hdu_spec.header["naxis1"]) - hdu_spec.header[
                "crpix1"]) * hdu_spec.header["cdelt1"] + hdu_spec.header["crval1"]

            # Clip the skylines:
            spec_data, mask = clip_skylines(
                wavelength, spec_data, spec_data_errs)

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
            norm_factors.append(norm_factor)

    # Pulls out just the flux values of each spectrum
    norm_interp_specs = [interp_cluster_spectra_dfs[i]['f_lambda_norm']
                         for i in range(len(interp_cluster_spectra_dfs))]

    nonzero_idxs = [np.nonzero(np.array(norm_interp_specs[i]))
                    for i in range(len(norm_interp_specs))]

    sum_norms = [np.zeros(len(norm_interp_specs[i]))
                 for i in range(len(norm_interp_specs))]

    for i in range(len(norm_interp_specs)):
        sum_norms[i][nonzero_idxs[i]] = norm_factors[i]
    # This sum_norms variable is a list of arrays. Each array correponds to
    # one spectrum from one galaxy. In that array, every point for which this
    # galaxy has a non-masked flux, there is it's normalization value. All
    # other points are zero

    # For each wavelength, counts the number of spectra that are nonzero - NOT
    # USING
    number_specs_by_wave = np.count_nonzero(norm_interp_specs, axis=0)

    norm_value_specs_by_wave = np.sum(sum_norms, axis=0)

    summed_spec = np.sum(norm_interp_specs, axis=0)

    # Have to use divz since lots of zeros
    total_spec = divz(summed_spec, norm_value_specs_by_wave)
    # Now we have divided each point by the sum of the normalizations that
    # contributed to it.
    total_spec_df = pd.DataFrame(zip(spectrum_wavelength, total_spec, number_specs_by_wave, norm_value_specs_by_wave),
                                 columns=['wavelength', 'f_lambda', 'n_galaxies', 'norm_value_summed'])

    total_spec_df.to_csv(
        imd.cluster_dir + f'/composite_spectra/{groupID}_spectrum.csv', index=False)

    plot_spec(groupID)
    return


def plot_spec(groupID):
    """Plots the spectrum

    Parameters:
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

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.09, 0.35, 0.88, 0.60])
    ax_Ha = fig.add_axes([0.69, 0.70, 0.25, 0.21])
    ax_contribute = fig.add_axes([0.09, 0.08, 0.88, 0.22])

    ax_Ha.spines['bottom'].set_color('red')
    ax_Ha.spines['top'].set_color('red')
    ax_Ha.spines['right'].set_color('red')
    ax_Ha.spines['left'].set_color('red')

    spectrum = total_spec_df['f_lambda']
    wavelength = total_spec_df['wavelength']
    n_galaxies = total_spec_df['n_galaxies']
    norm_value_summed = total_spec_df['norm_value_summed']

    wave_bin, spec_bin = median_bin_spec(wavelength, spectrum)

    ax.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
    ax.plot(wave_bin, spec_bin, color='orange', lw=1, label='Median Binned')
    ax_Ha.plot(wavelength, spectrum, color='black', lw=1)
    ax_contribute.plot(wavelength, n_galaxies, color='orange',
                       lw=1, label='Number of Galaxies')
    ax_contribute.plot(wavelength, norm_value_summed, color='black',
                       lw=1, label='Normalized Value of Galaxies')

    ax.set_ylim(-1 * 10**-18, 1.01 * np.max(spectrum))
    y_Ha_lim_max = np.max(spectrum[np.logical_and(
        wavelength > 6570, wavelength < 6800)])
    y_Ha_lim_min = np.min(spectrum[np.logical_and(
        wavelength > 6570, wavelength < 6800)])
    ax_Ha.set_ylim(y_Ha_lim_min, y_Ha_lim_max * 1.1)
    ax_Ha.set_xlim(6500, 6800)
    ax.legend(loc=2, fontsize=axisfont - 3)
    ax_contribute.legend(fontsize=axisfont - 3)

    rect = patches.Rectangle((6500, y_Ha_lim_min), 300, (y_Ha_lim_max -
                                                         y_Ha_lim_min), linewidth=1.5, edgecolor='red', facecolor='None')

    ax.add_patch(rect)
    # ax.set_xlim()
    ax_contribute.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax_contribute.set_ylabel('N', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax_contribute.tick_params(labelsize=ticksize, size=ticks)

    fig.savefig(imd.cluster_dir + f'/composite_spectra/{groupID}_spectrum.pdf')
    plt.close()


def divz(X, Y):
    return X / np.where(Y, Y, Y + 1) * np.not_equal(Y, 0)


def stack_all_spectra(n_clusters):
    """Runs the stack_spectra() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    for i in range(n_clusters):
        stack_spectra(i)


def plot_all_spectra(n_clusters):
    """Runs the plot_spec() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    for i in range(n_clusters):
        plot_spec(i)
