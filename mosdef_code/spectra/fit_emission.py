# Codes for simultaneously fitting the emission lines in a spectrum

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
from scipy.optimize import curve_fit
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from spectra_funcs import read_composite_spectrum, get_too_low_gals
import matplotlib.patches as patches


line_list = [
    ('Halpha', 6564.61),
    ('Hbeta', 4862.68),
    ('O3_5008', 5008.24),
    ('O3_4960', 4960.295),
    ('N2_6550', 6549.86),
    ('N2_6585', 6585.27)
]


def fit_emission(groupID, norm_method):
    """Given a groupID, fit the emission lines in that composite spectrum

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored

    Returns:
    Saves a csv of the fits for all of the lines
    """

    composite_spectrum_df = read_composite_spectrum(groupID, norm_method)

    # Build the initial guesses
    guess = []
    amp_guess = 10**-18  # flux units
    velocity_guess = 200  # km/s
    z_offset_guess = 0  # Angstrom
    continuum_offset_guess = 10**-20  # flux untis,
    # should eventually divide out the shape of continuum
    # First, we guess the z_offset
    guess.append(z_offset_guess)
    # THen, the continuum offset
    guess.append(continuum_offset_guess)
    # THen, the velocity
    guess.append(velocity_guess)
    # Then, for each line, we guess an amplitude
    for i in range(len(line_list)):
        guess.append(amp_guess)
    popt, pcov = curve_fit(auto_multi_gaussian, composite_spectrum_df[
        'wavelength'], composite_spectrum_df['f_lambda'], guess)

    # Now, parse the results into a dataframe
    line_names = [line_list[i][0] for i in range(len(line_list))]
    line_centers_rest = [line_list[i][1] for i in range(len(line_list))]
    z_offset = [popt[0] for i in range(len(line_list))]
    continuum_offset = [popt[1] for i in range(len(line_list))]
    velocity = [popt[2] for i in range(len(line_list))]
    sigs = [velocity_to_sig(line_list[i][1], popt[2])
            for i in range(len(line_list))]
    amps = popt[3:]

    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, continuum_offset, velocity, amps, sigs), columns=['line_name', 'line_center_rest', 'z_offset', 'continuum_offset', 'fixed_velocity', 'amplitude', 'sigma'])

    fit_df.to_csv(imd.cluster_dir + f'/emission_fitting/{groupID}_emission_fits.csv', index=False)
    plot_emission_fit(groupID, norm_method)
    return


def fit_all_emission(n_clusters, norm_method):
    """Runs the stack_spectra() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters
    norm_method (str): method of normalizing

    Returns:
    """
    for i in range(n_clusters):
        fit_emission(i, norm_method)


def plot_emission_fit(groupID, norm_method):
    """Plots the fit to each emission line

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored

    Returns:
    Saves a pdf of the fits for all of the lines
    """
    fit_df = ascii.read(imd.cluster_dir + f'/emission_fitting/{groupID}_emission_fits.csv').to_pandas()

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    total_spec_df = read_composite_spectrum(groupID, norm_method)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.09, 0.08, 0.88, 0.42])
    ax_Ha = fig.add_axes([0.55, 0.55, 0.40, 0.40])
    ax_Hb = fig.add_axes([0.09, 0.55, 0.40, 0.40])
    axes_arr = [ax, ax_Ha, ax_Hb]

    Ha_zoom_box_color = 'blue'
    ax_Ha.spines['bottom'].set_color(Ha_zoom_box_color)
    ax_Ha.spines['top'].set_color(Ha_zoom_box_color)
    ax_Ha.spines['right'].set_color(Ha_zoom_box_color)
    ax_Ha.spines['left'].set_color(Ha_zoom_box_color)

    Hb_zoom_box_color = 'violet'
    ax_Hb.spines['bottom'].set_color(Hb_zoom_box_color)
    ax_Hb.spines['top'].set_color(Hb_zoom_box_color)
    ax_Hb.spines['right'].set_color(Hb_zoom_box_color)
    ax_Hb.spines['left'].set_color(Hb_zoom_box_color)

    spectrum = total_spec_df['f_lambda']
    wavelength = total_spec_df['wavelength']
    n_galaxies = total_spec_df['n_galaxies']
    norm_value_summed = total_spec_df['norm_value_summed']

    too_low_gals, plot_cut, not_plot_cut, n_gals_in_group, cutoff, cutoff_low, cutoff_high = get_too_low_gals(
        groupID, norm_method)

    # Set up the parameters from the fitting
    pars = []
    pars.append(fit_df['z_offset'].iloc[0])
    pars.append(fit_df['continuum_offset'].iloc[0])
    pars.append(fit_df['fixed_velocity'].iloc[0])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['amplitude'])
    gauss_fit = auto_multi_gaussian(wavelength, pars)

    # Plots the spectrum and fit on all axes
    for axis in axes_arr:
        axis.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
        axis.plot(wavelength, gauss_fit, color='orange',
                  lw=1, label='Gaussian Fit')
        if axis != ax:
            # Add text for each of the lines:
            for i in range(len(line_list)):
                line_name = line_list[i][0]
                line_wave = line_list[i][1] + fit_df['z_offset'].iloc[0]
                line_idxs = np.logical_and(
                    wavelength > line_wave - 10, wavelength < line_wave + 10)
                axis.text(
                    line_wave - 10, np.max(spectrum[line_idxs]) * 1.02, line_name, fontsize=10)
                axis.plot([line_wave, line_wave], [-100, 100],
                          ls='--', alpha=0.5, color='mediumseagreen')

    # Plots the region with too few galaxies on the main axis
    ax.plot(wavelength[too_low_gals][plot_cut], spectrum[too_low_gals][plot_cut],
            color='red', lw=1, label=f'Too Few Galaxies ({cutoff})')
    ax.plot(wavelength[too_low_gals][not_plot_cut], spectrum[too_low_gals][not_plot_cut],
            color='red', lw=1)

    #ax.set_ylim(-1 * 10**-20, 1.01 * np.max(spectrum))
    ax.set_ylim(-1 * 10**-20, 8 * np.median(spectrum))
    Ha_plot_range = (6530, 6600)  # Angstrom
    Hb_plot_range = (4840, 5030)

    def set_plot_ranges(ax, axis, plot_range, box_color):
        lim_min = 0.9 * np.min(spectrum[np.logical_and(
            wavelength > plot_range[0], wavelength < plot_range[1])])
        lim_max = 1.05 * np.max(spectrum[np.logical_and(
            wavelength > plot_range[0], wavelength < plot_range[1])])
        axis.set_ylim(lim_min, lim_max)
        axis.set_xlim(plot_range)
        rect = patches.Rectangle((plot_range[0], lim_min), (plot_range[
                                 1] - plot_range[0]), (lim_max - lim_min), linewidth=1.5, edgecolor=box_color, facecolor='None')
        ax.add_patch(rect)

    set_plot_ranges(ax, ax_Ha, Ha_plot_range, Ha_zoom_box_color)
    set_plot_ranges(ax, ax_Hb, Hb_plot_range, Hb_zoom_box_color)

    ax.legend(loc=1, fontsize=axisfont - 3)

    ax.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    fig.savefig(imd.cluster_dir + f'/emission_fitting/{groupID}_emission_fit.pdf')
    plt.close()
    return


def gaussian_func(wavelength, peak_wavelength, amp, sig):
    """Standard Gaussian funciton

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    peak_wavelength (float): Peak of the line in the rest frame [angstroms]
    amp (float): Amplitude of the Gaussian
    sig (float): Standard deviation of the gaussian [angstroms]

    Returns:
    """
    return amp * np.exp(-(wavelength - peak_wavelength)**2 / (2 * sig**2))


def multi_gaussian(wavelength, *pars):
    """Fits all Gaussians simulatneously at fixed redshift

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    pars (list): List of all of the parameters

    Returns:
    """
    if len(pars) == 1:
        pars = pars[0]
    z_offset = pars[0]
    offset = pars[1]
    velocity = pars[2]
    g0 = gaussian_func(wavelength, line_list[0][
        1] + z_offset, pars[3], velocity_to_sig(line_list[0][1], velocity))
    g1 = gaussian_func(wavelength, line_list[1][
        1] + z_offset, pars[4],  velocity_to_sig(line_list[1][1], velocity))
    g2 = gaussian_func(wavelength, line_list[2][
        1] + z_offset, pars[5], velocity_to_sig(line_list[2][1], velocity))
    g3 = gaussian_func(wavelength, line_list[3][
        1] + z_offset, pars[6], velocity_to_sig(line_list[3][1], velocity))
    g4 = gaussian_func(wavelength, line_list[4][
        1] + z_offset, pars[7], velocity_to_sig(line_list[4][1], velocity))
    g5 = gaussian_func(wavelength, line_list[5][
        1] + z_offset, pars[8], velocity_to_sig(line_list[5][1], velocity))
    return g0 + g1 + g2 + g3 + g4 + g5 + offset


def auto_multi_gaussian(wavelength, *pars):
    """Fits all Gaussians simulatneously at fixed redshift

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    pars (list): List of all of the parameters

    Returns:
    """
    if len(pars) == 1:
        pars = pars[0]
    z_offset = pars[0]
    offset = pars[1]
    velocity = pars[2]
    gaussians = []
    for i in range(len(line_list)):
        gaussian = gaussian_func(wavelength, line_list[i][
                                 1] + z_offset, pars[i + 3], velocity_to_sig(line_list[i][1], velocity))
        gaussians.append(gaussian)

    return np.sum(gaussians, axis=0) + offset


def velocity_to_sig(line_center, velocity):
    '''Given line center and velocity, get the std deviation of the Gaussian

    Parameters:
    line_center (float): Central wavelength of line (angstrom)
    velocity (float): Rotational velocity of the galaxy (km/s). There might be a factor of 2 or so missing here

    Returns:
    sig (float): Standard deviation of the gaussian (angstrom)

    '''
    sig = line_center * (velocity / (3 * 10**5))
    return sig
