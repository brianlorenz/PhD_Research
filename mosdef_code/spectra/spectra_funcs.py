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
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
import mpu
from scipy import interpolate


def clip_skylines(wavelength, spectrum, spectrum_errs, mask_negatives=False):
    """Automated way to remove skylines form a spectrum

    Parameters:
    wavelength (array): array of the wavelength range
    spectrum (array): array of the corresponding f_lambda values
    spectrum_errs (array): array of the corresponding f_lambda uncertainty values
    mask_negatives (boolean): Set to true to mask all negative values


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

    thresh = 10 * np.median(spectrum_errs)
    for i in range(len(spectrum)):
        if spectrum_errs[i] > thresh:
            # Masks one pixel on either side of the current pixel
            mask[i - 1:i + 2] = 0
    '''
    thresh = 3
    sig_noise = divz(spectrum, spectrum_errs)
    mask = sig_noise > thresh
    '''

    if mask_negatives:
        for i in range(len(spectrum)):
            if spectrum[i] < 0:
                mask[i] = 0

    spectrum_clip = spectrum * mask
    err_spectrum_clip = spectrum_errs * mask

    return spectrum_clip, mask, err_spectrum_clip


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
    spectrum (array): pd.DatatFrame of the corresponding f_lambda values
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
        wave_range = wavelength[count_idx:count_idx + binsize]
        spec_range = spectrum[count_idx:count_idx + binsize]
        wave_bin.append(np.median(wavelength[count_idx:count_idx + binsize]))
        spec_ne_zero = spec_range[spec_range != 0]
        if len(spec_ne_zero) > 1:
            spec_bin.append(np.median(spec_ne_zero))
        else:
            spec_bin.append(0)
        count_idx = count_idx + binsize
    wave_bin = np.array(wave_bin)
    spec_bin = np.array(spec_bin)
    return wave_bin, spec_bin


def smooth_spectrum(spectrum, width=25):
    """Smooths a spectrum by setting each point to the median of surrounding points

    Parameters:
    spectrum (pd.DataFrame): pd.DatatFrame of the corresponding f_lambda values

    Returns:
    smooth_spec (array): Binned spectral value
    """
    smooth_spec = np.zeros(spectrum.shape)
    # repeat values near the edges of the spectrum for better medianing
    spectrum_pad = np.pad(spectrum, width, 'constant',
                          constant_values=(spectrum.iloc[0], spectrum.iloc[-1]))
    for i in range(len(smooth_spec)):
        j = i + width
        spec_range_values = spectrum_pad[j - width:j + width]
        nonzero = spec_range_values != 0
        nonzero_values = spec_range_values[nonzero]
        if len(nonzero_values) < 1:
            nonzero_values = -99
        smooth_spec[i] = np.median(nonzero_values)
    return smooth_spec


def read_composite_spectrum(groupID, norm_method):
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


def read_axis_ratio_spectrum(axis_group):
    """Reads in the spectrum file for a given cluster

    Parameters:
    axis_group (int): id of the group to read

    Returns:
    spectrum_df (pd.DataFrame): Dataframe containing wavelength and fluxes for the spectrum
    """
    spectrum_df = ascii.read(
        imd.cluster_dir + f'/composite_spectra/axis_stack/{axis_group}_spectrum.csv').to_pandas()

    return spectrum_df


def norm_spec_sed(composite_sed, spectrum_df):
    """Gets the normalization and correlation between composite SED and composite spectrum

    Parameters:
    composite_sed (pd.DataFrame): From read_composite_sed
    spectrum_df (pd.DataFrame): From stack_spectra.py, read in spectra then clipped skylines

    Returns:
    a12 (float): Normalization coefficient
    b12 (float): correlation, where 0 is identical and 1 is uncorrelated
    used_fluxes_df (pd.DataFrame): DataFrame containing the wavelengths and fluxes of the points that were compared on
    """
    # First, we find the points in the composite SED that we can correlate against
    # Want to use all points not near edges of spectrum
    edge = 20
    smooth_width = 200

    masked_spectrum_df = spectrum_df[spectrum_df['mask'] == 1]
    spectrum_flux = masked_spectrum_df['f_lambda']
    spectrum_wavelength = masked_spectrum_df['rest_wavelength']

    min_wave = spectrum_wavelength.iloc[edge]
    max_wave = spectrum_wavelength.iloc[-edge]

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

    f1 = sed_flux[sed_idxs[0]]
    f2 = np.array(interp_binned_spec(compare_waves))

    a12 = divz(np.sum(f1 * f2), np.sum(f2**2))
    b12 = np.sqrt(np.sum((f1 - a12 * f2)**2) / np.sum(f1**2))

    print(f'Normalization: {a12}')

    used_fluxes_df = pd.DataFrame(zip(compare_waves, f1, f2), columns=[
                                  'wavelength', 'sed_flux', 'spectrum_flux'])
    return a12, b12, used_fluxes_df


def get_too_low_gals(groupID, norm_method, thresh=0.15, axis_group=-1):
    """Given a groupID, find out which parts of the spectrum have too few galaxies to be useable

    Parameters:
    groupID (int): ID of the cluster to use
    norm_method (str): Normalization method used
    thresh (float): from 0 to 1, fraction of galaxies over which is acceptable. i.e., thresh=0.1 means to good parts of the spectrum have at least 10% of the number of galaxies in the cluster
    axis_group (int): Set to greater than -1 if you are plotting axis ratios instead

    Returns:
    too_low_gals (pd.DataFrame): True/False frame of where the spectrum is 'good'
    plot_cut (pd.DataFrame): Frist half of the above frame, less than 500 angstroms. Used for plotting
    not_plot_clut (pd.DataFrame): Other half of the above frame, used for plotting
    n_gals_in_group (int): Number of galaxies in the cluster
    cutoff (int): Number of galaixes above which ist acceptable
    """
    if axis_group > -1:
        total_spec_df = read_axis_ratio_spectrum(axis_group)
        ar_df = ascii.read(
            imd.cluster_dir + f'/composite_spectra/axis_stack/{axis_group}_df.csv').to_pandas()
        n_gals_in_group = len(ar_df)
    else:
        n_gals_in_group = len(os.listdir(imd.cluster_dir + '/' + str(groupID)))
        total_spec_df = read_composite_spectrum(groupID, norm_method)
    wavelength = total_spec_df['wavelength']
    n_galaxies = total_spec_df['n_galaxies']
    # too_low_gals = (n_galaxies / n_gals_in_group) < thresh
    too_low_gals = (n_galaxies / np.max(n_galaxies)) < thresh
    plot_cut = (wavelength[too_low_gals] > 5000)
    not_plot_cut = np.logical_not(plot_cut)
    cutoff = int(thresh * np.max(n_galaxies))
    cut_wave_low = np.percentile(wavelength[too_low_gals][not_plot_cut], 95)
    cut_wave_high = np.percentile(wavelength[too_low_gals][plot_cut], 5)

    return too_low_gals, plot_cut, not_plot_cut, n_gals_in_group, cutoff, cut_wave_high, cut_wave_low


def prepare_mock_observe_spectrum(groupID):
    """Calculates filter ranges for all SEDs in the group

    Parameters:
    groupID (int): ID of the cluster

    Returns:
    filter_dfs(list of pd.DataFrames): Each entry of the list is the a dataframe of a filter curve
    bounds (list of tuples of floats): (start_wavelength, end_wavelength) for each filter curve
    points (list of ints): Wavelength of each point that the filter corresponds to on the composite sed
    """
    filter_dir = imd.home_dir + f'/mosdef/composite_sed_csvs/composite_filter_csvs/{groupID}_filter_csvs/'
    filter_files = os.listdir(filter_dir)
    filter_dfs = [ascii.read(filter_dir + file).to_pandas()
                  for file in filter_files]
    filt_starts = [filter_df.iloc[filter_df[filter_df.transmission.ne(
        0)].index[0]].rest_wavelength for filter_df in filter_dfs]
    filt_ends = [filter_df.iloc[filter_df[filter_df.transmission.ne(
        0)].index[-1]].rest_wavelength for filter_df in filter_dfs]
    bounds = list(zip(filt_starts, filt_ends))
    points = [int(file[6:-4]) for file in filter_files]
    return filter_dfs, bounds, points


def mock_observe_spectrum(composite_sed, spectrum_df, filter_dfs, bounds, points):
    """Calculates filter ranges for all SEDs in the group

    Parameters:
    composite_sed (pd.DataFrame): Composite SED of the group, from read_composite_sed()
    spectrum_df (pd.DataFrame): from stack_spectra.py, read in spectrum with sky lines clipped
    filter_dfs(list of pd.DataFrames): Each entry of the list is the a dataframe of a filter curve
    bounds (list of tuples of floats): (start_wavelength, end_wavelength) for each filter curve
    points (list): Centers of the points

    Returns:
    scaled_flux_filter_nu (float): The result of observing the specturm in this filter, scaled by the amount of flux missing
    fraction_in_range (float): Percentage of the filter transmission that the spectrum covers - 1 means the spectrum is fully inside the filter. Errors increase as this gets lower
    composite_wave (float): wavelength used for comparison with composite SED
    composite_flux (float): flux value at that wavelength

    """
    masked_spectrum_df = spectrum_df[spectrum_df['mask'] == 1]

    spectrum_flux_clip = masked_spectrum_df['f_lambda_clip']
    spectrum_wavelength = masked_spectrum_df['rest_wavelength']

    spectrum_wavelength_min = spectrum_wavelength.iloc[0]
    spectrum_wavelength_max = spectrum_wavelength.iloc[-1]
    spectrum_wavelength_mid = np.median(spectrum_wavelength)

    # Then, use the one that is the most centered on the spectrum
    idx_use = np.argmin(np.abs(points - spectrum_wavelength_mid))

    filter_df = filter_dfs[idx_use]

    # Observes the filter
    interp_spec = interpolate.interp1d(spectrum_wavelength, spectrum_flux_clip)
    interp_filt = interpolate.interp1d(
        filter_df['rest_wavelength'], filter_df['transmission'])
    numerator = integrate.quad(lambda wave: (1 / 3**18) * (wave * interp_spec(wave) *
                                                           interp_filt(wave)), spectrum_wavelength_min, spectrum_wavelength_max)[0]
    denominator = integrate.quad(lambda wave: (
        interp_filt(wave) / wave), spectrum_wavelength_min, spectrum_wavelength_max)[0]
    flux_filter_nu = numerator / denominator

    # We need to correct for the fraction of missing flux - BIG ASSUMPTION
    # HERE
    transmission_spectrum_range = integrate.quad(lambda wave: (interp_filt(
        wave)), spectrum_wavelength_min, spectrum_wavelength_max)[0]
    transmission_full_range = integrate.quad(lambda wave: (interp_filt(
        wave)), filter_df['rest_wavelength'].iloc[0], filter_df['rest_wavelength'].iloc[-1])[0]
    fraction_in_range = transmission_spectrum_range / transmission_full_range

    scaled_flux_filter_nu = flux_filter_nu / fraction_in_range

    composite_sed_idx = np.argmin(
        np.abs(composite_sed['rest_wavelength'] - points[idx_use]))

    composite_wave = composite_sed['rest_wavelength'].iloc[composite_sed_idx]
    composite_flux = composite_sed['f_lambda'].iloc[composite_sed_idx]

    return scaled_flux_filter_nu, fraction_in_range, composite_wave, composite_flux


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


def read_spectrum(mosdef_obj, spectrum_file):
    """Reads a spectrum, returning fluxes, wavelength, and flux errors

    Parameters:
    mosdef_obj (pd.DataFrame): Run the get_mosdef_obj() function for this
    spectrum_file (str): Name of the spectrum

    Returns:
    spectrum_df (pd.DataFrame): Dataframe containing spectrum wavlength, fluxes, and uncertainties

    """
    z_spec = mosdef_obj['Z_MOSFIRE']
    field = mosdef_obj['FIELD_STR']
    v4id = mosdef_obj['V4ID']

    spec_loc = imd.spectra_dir + '/' + spectrum_file
    hdu_spec = fits.open(spec_loc)[1]
    hdu_errs = fits.open(spec_loc)[2]
    spec_data = hdu_spec.data
    spec_data_errs = hdu_errs.data

    # Compute the wavelength range
    wavelength = (1. + np.arange(hdu_spec.header["naxis1"]) - hdu_spec.header[
        "crpix1"]) * hdu_spec.header["cdelt1"] + hdu_spec.header["crval1"]

    rest_wavelength = wavelength / (1 + z_spec)

    spectrum_df = pd.DataFrame(zip(rest_wavelength, spec_data, spec_data_errs, wavelength), columns=[
                               'rest_wavelength', 'f_lambda', 'err_f_lambda', 'obs_wavelength'])

    return spectrum_df


def check_line_coverage(mosdef_obj, plot=False):
    """Checks to see if all five emission lines fall within the spectra for this object

    Parameters:
    mosdef_obj (pd.DataFrame): From get_mosdef_obj(field, v4id)
    plot (boolean): Set to True if you want to plot the spectra and emission lines

    Returns:
    """
    # Number of angstroms around the line that need to not be masked on either
    # side
    check_range = 4

    line_list = [
        ('Halpha', 6564.61),
        ('Hbeta', 4862.68),
        ('O3_5008', 5008.24),
        ('O3_4960', 4960.295),
        ('N2_6585', 6585.27)
    ]
    spectra_files = get_spectra_files(mosdef_obj)
    spectrum_dfs = []
    for file in spectra_files:
        spectrum_df = read_spectrum(mosdef_obj, file)
        spectrum_df['f_lambda_clip'], spectrum_df['mask'], spectrum_df['err_f_lambda_clip'] = clip_skylines(
            spectrum_df['obs_wavelength'], spectrum_df['f_lambda'], spectrum_df['err_f_lambda'])
        spectrum_dfs.append(spectrum_df)

    lines_ok = []
    for line in line_list:
        # Assume the line is bad
        line_ok = 0
        coverage = 0

        line_name = line[0]
        line_wave = line[1]
        for spectrum_df in spectrum_dfs:
            # Check if the line is even covered. If not, it will loop to the
            # next dataframe
            if line_wave > spectrum_df['rest_wavelength'].iloc[0] and line_wave < spectrum_df['rest_wavelength'].iloc[-1]:
                # If the line is covered, then we need to check if it is near
                # bad pixels
                coverage = 1

                # Find the index nearest to the line
                index = np.argmin(
                    np.abs(spectrum_df['rest_wavelength'] - line_wave))
                low_idx = np.max([0, index - check_range])
                high_idx = np.min([len(spectrum_df), index + check_range])
                if 0 not in spectrum_df.iloc[low_idx:high_idx]['f_lambda_clip'].to_numpy():
                    line_ok = 1
                    print(f'{line_name} is ok')
                else:
                    print(spectrum_df.iloc[low_idx:high_idx]['f_lambda_clip'])
                    print(f'{line_name} has bad pixels nearby')
        if coverage == 0:
            print(f'{line_name} has no coverage')
        # After checking all of the dataframes, append whether of not the line
        # is ok
        lines_ok.append(line_ok)
    if 0 in lines_ok:
        all_ok = False
        print(f"{mosdef_obj['FIELD_STR']}, {mosdef_obj['V4ID']} has at least one bad line")
    else:
        all_ok = True
        print(f"{mosdef_obj['FIELD_STR']}, {mosdef_obj['V4ID']} has full coverage!")
    if plot:
        plot_coverage(spectrum_dfs, line_list, check_range, lines_ok)
    return all_ok


def divz(X, Y):
    return X / np.where(Y, Y, Y + 1) * np.not_equal(Y, 0)


def plot_coverage(spectrum_dfs, line_list, check_range, lines_ok):
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for spectrum_df in spectrum_dfs:
        ax.plot(spectrum_df['rest_wavelength'], spectrum_df[
                'f_lambda'], color='blue', lw=1, alpha=0.5)
        ax.plot(spectrum_df['rest_wavelength'], spectrum_df[
                'f_lambda_clip'], color='black', lw=1)
    for i in range(len(line_list)):
        line = line_list[i]
        line_name = line[0]
        line_wave = line[1]
        ax.plot([line_wave, line_wave], [-100, 100], color='purple', lw=1)
        if lines_ok[i] == 1:
            ok_color = 'green'
        else:
            ok_color = 'red'
        ax.axvspan(line_wave - check_range, line_wave +
                   check_range, facecolor=ok_color, alpha=0.5)

    ax.set_ylim(np.percentile(spectrum_df[
                'f_lambda'], 1), np.percentile(spectrum_df[
                    'f_lambda'], 99))

    plt.show()
    # ax.set_xlim()

    # fig.savefig(imd.cluster_dir +
    # f'/composite_spectra/{groupID}_spectrum.pdf')
