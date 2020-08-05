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
from scipy.optimize import curve_fit, minimize, leastsq
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from spectra_funcs import read_axis_ratio_spectrum, read_composite_spectrum, get_too_low_gals
import matplotlib.patches as patches


line_list = [
    ('Halpha', 6564.61),
    ('Hbeta', 4862.68),
    ('O3_5008', 5008.24),
    ('O3_4960', 4960.295),
    ('N2_6550', 6549.86),
    ('N2_6585', 6585.27)
]


def fit_emission(groupID, norm_method, constrain_O3=False, axis_group=-1):
    """Given a groupID, fit the emission lines in that composite spectrum

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    constrain_O3 (boolean): Set to True to constrain the fitting of O3 to have a flux ratio of 2.97
    axis_group (int): Set to the number of the axis ratio group to fit that instead

    Returns:
    Saves a csv of the fits for all of the lines
    """

    if axis_group > -1:
        composite_spectrum_df = read_axis_ratio_spectrum(axis_group)
    else:
        composite_spectrum_df = read_composite_spectrum(groupID, norm_method)

    line_names = [line_list[i][0] for i in range(len(line_list))]

    # Build the initial guesses
    guess = []
    bounds_low = []
    bounds_high = []
    amp_guess = 10**-18  # flux units
    velocity_guess = 200  # km/s
    z_offset_guess = 0  # Angstrom
    continuum_offset_guess = 10**-20  # flux untis,
    slope_guess = 0  # flux untis,
    # should eventually divide out the shape of continuum
    # First, we guess the z_offset
    guess.append(z_offset_guess)
    # THen, the continuum offset
    guess.append(continuum_offset_guess)
    # THen, the velocity
    guess.append(velocity_guess)
    # Then, for each line, we guess an amplitude
    for i in range(len(line_list)):
        if 'O3_5008' in line_names and 'O3_4960' in line_names:
            idx_5008 = line_names.index('O3_5008')
            idx_4960 = line_names.index('O3_4960')
            if i == idx_5008:
                guess.append(1)
                continue
        guess.append(amp_guess)
    guess.append(slope_guess)

    n_loops = 10
    breakpoint()
    monte_carlo_fit(multi_gaussian, composite_spectrum_df[
        'wavelength'], composite_spectrum_df['f_lambda'], composite_spectrum_df['err_f_lambda'], guess, n_loops)
    # popt, pcov = curve_fit(multi_gaussian, composite_spectrum_df[
    #     'wavelength'], composite_spectrum_df['f_lambda'], guess)

    # Now, parse the results into a dataframe
    line_names = [line_list[i][0] for i in range(len(line_list))]
    line_centers_rest = [line_list[i][1] for i in range(len(line_list))]
    z_offset = [popt[0] for i in range(len(line_list))]
    continuum_offset = [popt[1] for i in range(len(line_list))]
    velocity = [popt[2] for i in range(len(line_list))]
    sigs = [velocity_to_sig(line_list[i][1], popt[2])
            for i in range(len(line_list))]

    if 'O3_5008' in line_names and 'O3_4960' in line_names:
        idx_5008 = line_names.index('O3_5008')
        idx_4960 = line_names.index('O3_4960')
        sig_5008 = sigs[idx_5008]
        sig_4960 = sigs[idx_4960]
        amp_4960 = popt[idx_4960 + 3]
        if constrain_O3 == True:
            amp_5008 = 2.97 * amp_4960 * (sig_4960 / sig_5008)
        else:
            amp_5008 = 2.97 * amp_4960 * \
                (sig_4960 / sig_5008) * popt[idx_5008 + 3]
        popt[3 + idx_5008] = amp_5008

    amps = popt[3:-1]
    slope = [popt[-1] for i in range(len(line_list))]
    fluxes = [get_flux(amps[i], sigs[i]) for i in range(len(line_list))]

    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, continuum_offset, velocity, amps, sigs, fluxes, slope), columns=['line_name', 'line_center_rest', 'z_offset', 'continuum_offset', 'fixed_velocity', 'amplitude', 'sigma', 'flux', 'slope'])

    if axis_group > -1:
        fit_df.to_csv(imd.cluster_dir + f'/emission_fitting/axis_ratio_clusters/{axis_group}_emission_fits.csv', index=False)
        plot_emission_fit(groupID, norm_method, axis_group=axis_group)
    else:
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


def plot_emission_fit(groupID, norm_method, axis_group=-1):
    """Plots the fit to each emission line

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    axis_group (int): Set to the number of the axis ratio group to fit that instead

    Returns:
    Saves a pdf of the fits for all of the lines
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    if axis_group > -1:
        fit_df = ascii.read(imd.cluster_dir + f'/emission_fitting/axis_ratio_clusters/{axis_group}_emission_fits.csv').to_pandas()
        total_spec_df = read_axis_ratio_spectrum(axis_group)
    else:
        fit_df = ascii.read(imd.cluster_dir + f'/emission_fitting/{groupID}_emission_fits.csv').to_pandas()
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
        groupID, norm_method, axis_group=axis_group)

    # Set up the parameters from the fitting
    pars = []
    pars.append(fit_df['z_offset'].iloc[0])
    pars.append(fit_df['continuum_offset'].iloc[0])
    pars.append(fit_df['fixed_velocity'].iloc[0])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['amplitude'])
    pars.append(fit_df['slope'].iloc[0])
    gauss_fit = multi_gaussian(wavelength, pars, fit=False)

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

    # ax.set_ylim(-1 * 10**-20, 1.01 * np.max(spectrum))
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

    if axis_group > -1:
        fig.savefig(imd.cluster_dir + f'/emission_fitting/axis_ratio_clusters/{axis_group}_emission_fit.pdf')
    else:
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


def multi_gaussian(wavelength, *pars, fit=True):
    """Fits all Gaussians simulatneously at fixed redshift

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    pars (list): List of all of the parameters
    fit (boolean): Set to True if fitting (ie amps are not constrained yet)

    Returns:
    """
    if len(pars) == 1:
        pars = pars[0]
    z_offset = pars[0]
    offset = pars[1]
    velocity = pars[2]
    slope = pars[-1]

    # Force the fluxes of the OIII lines to be in the ratio 2.97
    line_names = [line_list[i][0] for i in range(len(line_list))]

    # If both lines are present, fix the ratio of the lines
    if 'O3_5008' in line_names and 'O3_4960' in line_names and fit == True:
        idx_5008 = line_names.index('O3_5008')
        idx_4960 = line_names.index('O3_4960')
        sig_5008 = velocity_to_sig(line_list[idx_5008][1], velocity)
        sig_4960 = velocity_to_sig(line_list[idx_4960][1], velocity)
        amp_4960 = pars[idx_4960 + 3]
        amp_5008 = 2.97 * amp_4960 * (sig_4960 / sig_5008) * pars[idx_5008 + 3]

    gaussians = []
    for i in range(len(line_list)):
        peak_wavelength = line_list[i][1]
        amp = pars[i + 3]
        # Special case to fix the ratio of O3 lines
        if line_list[i][0] == 'O3_5008' and fit == True:
            gaussian = gaussian_func(wavelength, peak_wavelength +
                                     z_offset, amp_5008, velocity_to_sig(peak_wavelength, velocity))
        else:
            gaussian = gaussian_func(wavelength, peak_wavelength +
                                     z_offset, amp, velocity_to_sig(peak_wavelength, velocity))
        gaussians.append(gaussian)
    return np.sum(gaussians, axis=0) + offset + slope * wavelength


def monte_carlo_fit(func, x_data, y_data, y_err, guess, n_loops):
    '''Fit the multi-gaussian to the data, use monte carlo to get uncertainties

    Parameters:
    x_data (pd.DataFrame): x data to be fit, 1d
    y_data (pd.DataFrame): y data to be fit, 1d
    y_err (pd.DataFrame): Uncertainties on the y_data
    guess (list): list of guesses for the parameters of the fit
    n_loops (int): Number of times to run the monte_carlo simulations

    Returns:
    popt (list): List of the fit parameters
    err_popt (list): Uncertainties on these parameters
    '''
    popt, pcov = curve_fit(func, x_data, y_data, guess)
    for i in range(n_loops):
        new_ys = [np.random_normal(loc=y_data.iloc[j], scale=y_err.iloc[
                                   j]) for j in range(len(y_data))]
        new_popt, new_pcov = curve_fit(func, x_data, new_ys, guess)
        if i == 0:
            arr_popt = np.array(new_popt)
        else:
            np.vstack((arr_popt, np.array(new_popt)))
    return


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


def get_flux(amp, sig):
    '''Given the amplitude and std deviation of a Gaussian, compute the line flux

    Parameters:
    amp (float): amplitude of gaussian (flux units)
    sig (float): Standard deviation of the gaussian (angstrom)

    Returns:
    flux (float): Total area under the Gaussian
    '''
    flux = amp * sig * np.sqrt(2 * np.pi)
    return flux


def fit_all_emission(n_clusters, norm_method, constrain_O3=False):
    """Runs the fit_emission() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters
    norm_method (str): Method of normalization

    Returns:
    """
    for i in range(n_clusters):
        print(f'Fitting emission for {i}')
        fit_emission(i, norm_method, constrain_O3=constrain_O3)
