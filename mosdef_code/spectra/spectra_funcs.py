# Functions taht deal with finding, reading, and modifying spectra

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
import mpu
from scipy import interpolate


def clip_skylines(wavelength, spectrum, spectrum_errs):
    """Automated way to remove skylines form a spectrum

    Parameters:
    wavelength (array): array of the wavelength range
    spectrum (array): array of the corresponding f_lambda values
    spectrum_errs (array): array of the corresponding f_lambda uncertainty values


    Returns:
    spectrum_clip (array): clipped spectrum, with skylines set to zero
    mask (array): 0 where the spectrum should be clipped, 1 where it is fine
    """

    '''
    # This clips based on high deviations from the median spectrum
    perc_low, perc_hi = np.percentile(spectrum, [2, 98])
    spectrum[spectrum > 2 * perc_hi] = 0.
    spectrum[spectrum < -np.abs(2 * perc_low)] = 0.
    '''

    thresh = np.percentile(spectrum, 95)
    mask = np.ones(len(spectrum))

    thresh = 3 * np.median(spectrum_errs)
    for i in range(len(spectrum)):
        if spectrum_errs[i] > thresh:
            # Masks one pixel on either side of the current pixel
            mask[i - 1:i + 2] = 0
    '''
    thresh = 3
    sig_noise = divz(spectrum, spectrum_errs)
    mask = sig_noise > thresh
    '''

    spectrum_clip = spectrum * mask

    return spectrum_clip, mask


def get_spectra_files(mosdef_obj, filt=0):
    """Finds the names of all of the spectra files for a given object, returning them as a list

    Parameters:
    mosdef_obj (pd.DataFrame): Get this through get_mosdef_obj
    filt (str): Set to the letter of the filter that you want files from

    Returns:
    obj_files (list): List of the filenames for the spectra of that object
    """
    all_spectra_files = os.listdir(imd.spectra_dir)
    obj_files = [filename for filename in all_spectra_files if f'.{mosdef_obj["ID"]}.ell' in filename]
    if filt:
        obj_files = [filename for filename in obj_files if f'{filt}' in filename]
    return obj_files


def median_bin_spec(wavelength, spectrum, binsize=10):
    """Median-bins a spectrum

    Parameters:
    wavelength (array): array of the wavelength range
    spectrum (array): array of the corresponding f_lambda values
    binsize (int): Number of points to bin over

    Returns:
    wave_bin (array): Binned wavelengths
    spec_bin (array): Binned spectral value
    """
    count_idx = 0
    wave_bin = []
    spec_bin = []
    while count_idx < len(wavelength):
        if count_idx + binsize > len(wavelength):
            binsize = len(wavelength) - count_idx
        wave_bin.append(np.median(wavelength[count_idx:count_idx + binsize]))
        spec_bin.append(np.median(spectrum[count_idx:count_idx + binsize]))
        count_idx = count_idx + binsize
    wave_bin = np.array(wave_bin)
    spec_bin = np.array(spec_bin)
    return wave_bin, spec_bin


def read_spectrum(groupID, norm_method='cluster_norm'):
    """Reads in the spectrum file for a given cluster

    Parameters:
    groupID (int): id of the cluster to read
    norm_method (str): folder to look for spectrum

    Returns:
    spectrum_df (pd.DataFrame): Dataframe containing wavelength and fluxes for the spectrum
    """
    spectrum_df = ascii.read(
        imd.cluster_dir + f'/composite_spectra/{norm_method}/{groupID}_spectrum.csv').to_pandas()

    return spectrum_df


def norm_spec_sed(composite_sed, spectrum_flux, spectrum_wavelength, mask):
    """Gets the normalization and correlation between composite SED and composite spectrum

    Parameters:
    composite_sed (pd.DataFrame): From read_composite_sed
    fluxes (pd.DataFrame): spectrum fluxes
    wavelength (array): spectrum wavelength
    mask (0,1 array): values of skylines to mask, 0 where spectrum should be clipped

    Returns:
    a12 (float): Normalization coefficient
    b12 (float): correlation, where 0 is identical and 1 is uncorrelated
    used_fluxes_df (pd.DataFrame): DataFrame containing the wavelengths and fluxes of the points that were compared on
    """
    # First, we find the points in the composite SED that we can correlate against
    # Want to use all points not near edges of spectrum
    edge = 50
    smooth_width = 200

    min_wave = spectrum_wavelength[edge]
    max_wave = spectrum_wavelength[-edge]

    wave_bin, spec_bin = median_bin_spec(
        spectrum_wavelength[edge:-edge], spectrum_flux[edge:-edge], binsize=smooth_width)
    interp_binned_spec = interpolate.interp1d(
        wave_bin, spec_bin, fill_value=0, bounds_error=False)

    sed_flux = composite_sed['f_lambda']
    sed_wavelength = composite_sed['rest_wavelength']

    sed_idxs = np.where(np.logical_and(
        sed_wavelength > min_wave, sed_wavelength < max_wave))

    compare_waves = sed_wavelength.iloc[sed_idxs]

    # Find where spectrum is closest to these values
    spectrum_idxs = [np.argmin(np.abs(spectrum_wavelength - wave))
                     for wave in compare_waves]

    # Meidan-smooth by smooth_width
    spectrum_fluxes = [np.median(spectrum_flux[idx - smooth_width:idx + smooth_width])
                       for idx in spectrum_idxs]

    f1 = sed_flux[sed_idxs[0]]
    #f2 = np.array(spectrum_fluxes)
    f2 = np.array(interp_binned_spec(compare_waves))

    a12 = divz(np.sum(f1 * f2), np.sum(f2**2))
    b12 = np.sqrt(np.sum((f1 - a12 * f2)**2) / np.sum(f1**2))

    print(f'Normalization: {a12}')

    used_fluxes_df = pd.DataFrame(zip(compare_waves, f1, f2), columns=[
                                  'wavelength', 'sed_flux', 'spectrum_flux'])
    return a12, b12, used_fluxes_df


def get_too_low_gals(groupID, thresh=0.1):
    """Given a groupID, find out which parts of the spectrum have too few galaxies to be useable

    Parameters:
    groupID (int): ID of the cluster to use
    thresh (float): from 0 to 1, fraction of galaxies over which is acceptable. i.e., thresh=0.1 means to good parts of the spectrum have at least 10% of the number of galaxies in the cluster

    Returns:
    too_low_gals (pd.DataFrame): True/False frame of where the spectrum is 'good'
    plot_cut (pd.DataFrame): Frist half of the above frame, less than 500 angstroms. Used for plotting
    not_plot_clut (pd.DataFrame): Other half of the above frame, used for plotting
    n_gals_in_group (int): Number of galaxies in the cluster
    cutoff (int): Number of galaixes above which ist acceptable
    """
    n_gals_in_group = len(os.listdir(imd.cluster_dir + '/' + str(groupID)))
    total_spec_df = read_spectrum(groupID)
    wavelength = total_spec_df['wavelength']
    n_galaxies = total_spec_df['n_galaxies']
    too_low_gals = (n_galaxies / n_gals_in_group) < thresh
    plot_cut = (wavelength[too_low_gals] > 5000)
    not_plot_cut = np.logical_not(plot_cut)
    cutoff = int(thresh * n_gals_in_group)
    cut_wave_low = np.percentile(wavelength[too_low_gals][not_plot_cut], 95)
    cut_wave_high = np.percentile(wavelength[too_low_gals][plot_cut], 5)

    return too_low_gals, plot_cut, not_plot_cut, n_gals_in_group, cutoff, cut_wave_high, cut_wave_low


def find_skylines(zobjs, filt):
    """Stacks many spectra to identify the birghtest skylines - Doesn't work

    Parameters:
    zobjs (list): Use get_zobjs() function. Only needs a slice of this
    filt (str): Set to the letter of the filter that you want files from e.g. 'H', J, K, Y

    Returns:
    """

    # Idea here is to read in a bunch of spectra (which will be at different
    # redshifts, then stack them, leaving only the skylines. Emission should
    # get washed out)

    # Doesn't Work, skylines are at different locations
    spectra_dfs = []

    for field, v4id in zobjs:
        if v4id < -99:
            continue
        mosdef_obj = get_mosdef_obj(field, v4id)
        spectra_files = get_spectra_files(mosdef_obj, filt)
        for spectrum_file in spectra_files:
            # Looping over each file, open the spectrum
            spec_loc = imd.spectra_dir + spectrum_file
            hdu = fits.open(spec_loc)[1]
            spec_data = hdu.data

            # Compute the wavelength range
            wavelength = (1. + np.arange(hdu.header["naxis1"]) - hdu.header[
                "crpix1"]) * hdu.header["cdelt1"] + hdu.header["crval1"]

            spectrum_df = pd.DataFrame(zip(wavelength, spec_data),
                                       columns=['obs_wavelength', 'f_lambda'])

            spectra_dfs.append(spectrum_df)

    fluxes = [spectra_dfs[i]['f_lambda']
              for i in range(len(spectra_dfs))]

    print(len(fluxes))

    fluxes_stack = np.sum(fluxes, axis=0)

    # Maybe need a more sophisticated way to cut here
    bad = np.greater(fluxes_stack, 50 * np.median(fluxes_stack)
                     * np.ones(len(fluxes_stack)))
    bad = np.logical_or(bad, np.less(fluxes_stack, -50 * np.median(fluxes_stack)
                                     * np.ones(len(fluxes_stack))))

    pd.DataFrame(bad).to_csv(imd.home_dir + f'/mosdef/Spectra/skyline_masks/{filt}_mask.csv', header=False, index=False)

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.plot(wavelength, fluxes_stack, color='black', lw=1)
    ax.scatter(wavelength[bad], fluxes_stack[bad], color='red')

    #ax.set_ylim(-1 * 10**-18, 1 * 10**-18)
    # ax.set_xlim()

    plt.tight_layout()
    # fig.savefig(imd.cluster_dir +
    # f'/composite_spectra/{groupID}_spectrum.pdf')
    plt.show()

    return


def divz(X, Y):
    return X / np.where(Y, Y, Y + 1) * np.not_equal(Y, 0)
