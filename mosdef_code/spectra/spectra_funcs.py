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


def clip_skylines(wavelength, spectrum, filt):
    """Automated way to remove skylines form a spectrum

    Parameters:
    wavelength (array): array of the wavelength range
    spectrum (array): array of the corresponding f_lambda values
    filt (str): letter of the filter. e.g. 'J'

    Returns:
    spectrum_clip (array): clipped spectrum, with skylines set to zero
    """

    # This stacks observed spectra to find the skylines, then masks them out
    filt_mask = ascii.read(imd.home_dir + f'/mosdef/Spectra/skyline_masks/{filt}_mask.csv', format='no_header').to_pandas()
    filt_mask.columns = ['mask']

    '''
    # This clips based on high deviations from the median spectrum
    perc_low, perc_hi = np.percentile(spectrum, [2, 98])
    spectrum[spectrum > 2 * perc_hi] = 0.
    spectrum[spectrum < -np.abs(2 * perc_low)] = 0.
    '''

    '''
    for skyline in skylines:
        bad = np.greater(wavelength, skyline - 3) * \
            np.less(wavelength, skyline + 3)
    '''

    mask = [mpu.string.str2bool(filt_mask['test'].iloc[i])
            for i in range(len(filt_mask))]

    spectrum[mask] = 0

    breakpoint()

    return spectrum


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


def find_skylines(zobjs, filt):
    """Stacks many spectra to identify the birghtest skylines

    Parameters:
    zobjs (list): Use get_zobjs() function. Only needs a slice of this
    filt (str): Set to the letter of the filter that you want files from e.g. 'H', J, K, Y

    Returns:
    """

    # Idea here is to read in a bunch of spectra (which will be at different
    # redshifts, then stack them, leaving only the skylines. Emission should
    # get washed out)
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
