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
    print(thresh)
    mask = np.ones(len(spectrum))

    thresh = 3 * np.median(spectrum_errs)
    for i in range(len(spectrum)):
        if spectrum_errs[i] > thresh:
            # Masks one pixel on either side of the current pixel
            mask[i - 1:i + 2] = 0

    thresh = 3
    sig_noise = spectrum / spectrum_errs
    mask = sig_noise < thresh

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
