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
import time

line_list = [
    ('Halpha', 6564.61),
    ('Hbeta', 4862.68),
    ('O3_5008', 5008.24),
    ('O3_4960', 4960.295),
    ('N2_6550', 6549.86),
    ('N2_6585', 6585.27)
]
line_centers_rest = [line_list[i][1] for i in range(len(line_list))]


def fit_emission(groupID, norm_method, constrain_O3=False, axis_group=-1, save_name='', scaled='False', run_name='False'):
    """Given a groupID, fit the emission lines in that composite spectrum

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    save_name(str): Folder of where to save and where spectra are located.
    constrain_O3 (boolean): Set to True to constrain the fitting of O3 to have a flux ratio of 2.97
    axis_group (int): Set to the number of the axis ratio group to fit that instead
    scaled (str): Set to 'True' if fitting the scaled spectra
    run_name (str): Set to the prospector run_name if fitting prospector spectra

    Returns:
    Saves a csv of the fits for all of the lines
    """

    if axis_group > -1:
        composite_spectrum_df = read_axis_ratio_spectrum(axis_group, save_name)
        fast_continuum_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv').to_pandas()
        fast_continuum = fast_continuum_df['f_lambda_scaled']
    elif scaled == 'True':
        composite_spectrum_df = read_composite_spectrum(
            groupID, norm_method, scaled='True')
    elif run_name != 'False':
        composite_spectrum_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_merged_spec.csv').to_pandas()
        composite_spectrum_df['err_f_lambda'] = (composite_spectrum_df['err_f_lambda_u'] + composite_spectrum_df['err_f_lambda_d'])/2   
        composite_spectrum_df['wavelength'] = composite_spectrum_df['rest_wavelength']
    else:
        composite_spectrum_df = read_composite_spectrum(groupID, norm_method)

    line_names = [line_list[i][0] for i in range(len(line_list))]

    # Scale factor to make fitting more robust
    if scaled == 'True':
        scale_factor = 10**15
    else:
        scale_factor = 10**18

    # Build the initial guesses
    guess = []
    bounds_low = []
    bounds_high = []
    amp_guess = 10**-18  # flux units
    if scaled == 'True':
        amp_guess = 10**-14  # flux units
    if run_name != 'False':
        amp_guess = 10**-15  # flux units
    velocity_guess = 200  # km/s
    z_offset_guess = 0  # Angstrom
    # continuum_offset_guess = 10**-20  # flux untis,
    continuum_offset_guess = 0.5  # Scale to multiply by FAST continuum,
    # should eventually divide out the shape of continuum
    # First, we guess the z_offset
    guess.append(z_offset_guess)
    bounds_low.append(-10)
    bounds_high.append(10)
    # THen, the velocity
    guess.append(velocity_guess)
    bounds_low.append(30)
    bounds_high.append(1000)
    # Then, for each line, we guess an amplitude
    for i in range(len(line_list)):
        # if 'O3_5008' in line_names and 'O3_4960' in line_names:
        #     idx_5008 = line_names.index('O3_5008')
        #     idx_4960 = line_names.index('O3_4960')
        #     if i == idx_5008:
        #         guess.append(1)
        #         continue
        guess.append(scale_factor * amp_guess)
        bounds_low.append(0)
        if scaled == 'True':
            bounds_high.append(scale_factor * 10**-12)
        elif run_name != 'False':
            bounds_high.append(scale_factor * 10**-12)
        else:
            bounds_high.append(scale_factor * 10**-16)
    bounds = (np.array(bounds_low), np.array(bounds_high))

    n_loops = 10
    wavelength = composite_spectrum_df[
        'wavelength']
    if scaled == 'True':
        continuum = composite_spectrum_df[
            'cont_f_lambda_scaled']
    else:
        continuum = composite_spectrum_df[
            'cont_f_lambda']
    full_cut = get_fit_range(wavelength)
    wavelength_cut = wavelength[full_cut]
    continuum_cut = continuum[full_cut]
    # print(bounds)
    # print(guess)
    if scaled == 'True':
        popt, arr_popt, cont_scale_out, y_data_cont_sub = monte_carlo_fit(multi_gaussian, wavelength_cut, scale_factor * continuum_cut, scale_factor * composite_spectrum_df[full_cut][
            'f_lambda_scaled'], scale_factor * composite_spectrum_df[full_cut]['err_f_lambda_scaled'], np.array(guess), bounds, n_loops)
    elif axis_group > -1:
        fast_continuum_cut = fast_continuum[full_cut]

        # Caveat for if the continuum is negative, then return to the old method
        if np.median(fast_continuum_cut) < 0:
            popt, arr_popt, cont_scale_out, y_data_cont_sub = monte_carlo_fit(multi_gaussian, wavelength_cut, scale_factor * continuum_cut, scale_factor * composite_spectrum_df[full_cut][
                'f_lambda'], scale_factor * composite_spectrum_df[full_cut]['err_f_lambda'], np.array(guess), bounds, n_loops)
        else:
            popt, arr_popt, cont_scale_out, y_data_cont_sub = monte_carlo_fit(multi_gaussian, wavelength_cut, scale_factor * continuum_cut, scale_factor * composite_spectrum_df[full_cut][
                'f_lambda'], scale_factor * composite_spectrum_df[full_cut]['err_f_lambda'], np.array(guess), bounds, n_loops, fit_axis_group=1, fast_continuum_cut=scale_factor*fast_continuum_cut
                )
    
    else:    
        popt, arr_popt, cont_scale_out, y_data_cont_sub = monte_carlo_fit(multi_gaussian, wavelength_cut, scale_factor * continuum_cut, scale_factor * composite_spectrum_df[full_cut][
            'f_lambda'], scale_factor * composite_spectrum_df[full_cut]['err_f_lambda'], np.array(guess), bounds, n_loops)
    err_popt = np.std(arr_popt, axis=0)
    # popt, pcov = curve_fit(multi_gaussian, composite_spectrum_df[
    #     'wavelength'], composite_spectrum_df['f_lambda'], guess)

    # Save the continuum-subtracted ydata
    if axis_group > -1:
        y_data_cont_sub = y_data_cont_sub / scale_factor
        cont_sub_df = pd.DataFrame(zip(wavelength_cut, y_data_cont_sub), columns=['wavelength_cut','continuum_sub_ydata'])
        cont_sub_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs/{axis_group}_cont_sub.csv', index=False)


    # Now, parse the results into a dataframe
    hb_scale, ha_scale, err_hb_scale, err_ha_scale = cont_scale_out
    hb_scales = [hb_scale for i in range(len(line_list))]
    ha_scales = [ha_scale for i in range(len(line_list))]
    err_hb_scales = [err_hb_scale for i in range(len(line_list))]
    err_ha_scales = [err_ha_scale for i in range(len(line_list))]
    line_names = [line_list[i][0] for i in range(len(line_list))]
    line_centers_rest = [line_list[i][1] for i in range(len(line_list))]
    z_offset = [popt[0] for i in range(len(line_list))]
    err_z_offset = [err_popt[0] for i in range(len(line_list))]
    velocity = [popt[1] for i in range(len(line_list))]
    err_velocity = [err_popt[1] for i in range(len(line_list))]
    sigs = [velocity_to_sig(line_list[i][1], popt[1])
            for i in range(len(line_list))]
    err_sigs = [velocity_to_sig(line_list[i][1], popt[1] + err_popt[1]) - sigs[i]
                for i in range(len(line_list))]

    amps = popt[2:] / scale_factor
    err_amps = err_popt[2:] / scale_factor
    flux_tuples = [get_flux(amps[i], sigs[i], amp_err=err_amps[i], sig_err=err_sigs[
                            i]) for i in range(len(line_list))]
    fluxes = [i[0] for i in flux_tuples]
    err_fluxes = [i[1] for i in flux_tuples]
    ha_idx = [idx for idx, name in enumerate(
        line_names) if name == 'Halpha'][0]
    hb_idx = [idx for idx, name in enumerate(line_names) if name == 'Hbeta'][0]
    ha_amps = [arr_popt[i][2 + ha_idx] for i in range(len(arr_popt))]
    hb_amps = [arr_popt[i][2 + hb_idx] for i in range(len(arr_popt))]
    ha_sigs = [velocity_to_sig(line_list[ha_idx][1], arr_popt[i][
                               1])for i in range(len(arr_popt))]
    hb_sigs = [velocity_to_sig(line_list[hb_idx][1], arr_popt[i][
                               1])for i in range(len(arr_popt))]
    all_ha_fluxes = [get_flux(ha_amps[i], ha_sigs[i])
                     for i in range(len(arr_popt))]
    all_hb_fluxes = [get_flux(hb_amps[i], hb_sigs[i])
                     for i in range(len(arr_popt))]
    all_balmer_decs = [all_ha_fluxes[i][0] / all_hb_fluxes[i][0]
                       for i in range(len(arr_popt))]
    balmer_dec = [fluxes[ha_idx] / fluxes[hb_idx]
                  for i in range(len(line_list))]

    err_balmer_dec_low = balmer_dec - np.percentile(all_balmer_decs, 16)
    err_balmer_dec_high = np.percentile(all_balmer_decs, 84) - balmer_dec

    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, err_z_offset, hb_scales, err_hb_scales, ha_scales, err_ha_scales, velocity, err_velocity, amps, err_amps, sigs, err_sigs, fluxes, err_fluxes, balmer_dec, err_balmer_dec_low, err_balmer_dec_high), columns=['line_name', 'line_center_rest', 'z_offset', 'err_z_offset', 'hb_scale', 'err_hb_scale', 'ha_scale', 'err_ha_scale', 'fixed_velocity', 'err_fixed_velocity', 'amplitude', 'err_amplitude', 'sigma', 'err_sigma', 'flux', 'err_flux', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high'])

    if axis_group > -1:
        fit_df.to_csv(
            imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv', index=False)
        plot_emission_fit(groupID, norm_method,
                          axis_group=axis_group, save_name=save_name)
    elif scaled == 'True':
        fit_df.to_csv(imd.emission_fit_csvs_dir +
                      f'/{groupID}_emission_fits_scaled.csv', index=False)
        plot_emission_fit(groupID, norm_method, scaled='True')
    elif run_name != 'False':
        imd.check_and_make_dir(imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits')
        fit_df.to_csv(imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits/{groupID}_emission_fits.csv', index=False)
        plot_emission_fit(groupID, norm_method, run_name=run_name)
    else:
        fit_df.to_csv(imd.emission_fit_csvs_dir +
                      f'/{groupID}_emission_fits.csv', index=False)
        plot_emission_fit(groupID, norm_method)
    return


def fit_all_emission(n_clusters, norm_method, scaled='False'):
    """Runs the stack_spectra() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters
    norm_method (str): method of normalizing
    scaled (str): Set to 'True' if fitting the scaled spectra instead

    Returns:
    """
    for i in range(n_clusters):
        fit_emission(i, norm_method, scaled=scaled)


def plot_emission_fit(groupID, norm_method, axis_group=-1, save_name='', scaled='False', run_name='False'):
    """Plots the fit to each emission line

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    axis_group (int): Set to the number of the axis ratio group to fit that instead
    scaled (str): Set to true if plotting the scaled fits
    run_name (str): Set to name of prospector run to fit with those

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
        fit_df = ascii.read(
            imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv').to_pandas()
        total_spec_df = read_axis_ratio_spectrum(axis_group, save_name)
        fast_continuum_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv').to_pandas()
        fast_continuum = fast_continuum_df['f_lambda_scaled']
        cont_sub_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs/{axis_group}_cont_sub.csv').to_pandas()
    elif scaled == 'True':
        fit_df = ascii.read(imd.emission_fit_csvs_dir +
                            f'/{groupID}_emission_fits_scaled.csv').to_pandas()
        total_spec_df = read_composite_spectrum(groupID, norm_method, scaled = 'True')
    elif run_name != 'False':
        fit_df = ascii.read(imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits/{groupID}_emission_fits.csv').to_pandas()
        total_spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_merged_spec.csv').to_pandas()
        total_spec_df['err_f_lambda'] = (total_spec_df['err_f_lambda_u'] + total_spec_df['err_f_lambda_d'])/2   
        total_spec_df['wavelength'] = total_spec_df['rest_wavelength']
    else:
        fit_df = ascii.read(imd.emission_fit_csvs_dir +
                            f'/{groupID}_emission_fits.csv').to_pandas()
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

    if scaled == 'True':
        spectrum = total_spec_df['f_lambda_scaled']
        continuum = total_spec_df['cont_f_lambda_scaled']
    else:
        spectrum = total_spec_df['f_lambda']
        continuum = total_spec_df['cont_f_lambda']
    wavelength = total_spec_df['wavelength']
    # n_galaxies = total_spec_df['n_galaxies']
    # norm_value_summed = total_spec_df['norm_value_summed']
    

    too_low_gals, plot_cut, not_plot_cut, n_gals_in_group, cutoff, cutoff_low, cutoff_high = get_too_low_gals(
        groupID, norm_method, save_name, axis_group=axis_group)

    # Set up the parameters from the fitting
    pars = []
    pars.append(fit_df['z_offset'].iloc[0])
    pars.append(fit_df['fixed_velocity'].iloc[0])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['amplitude'])

    full_cut = get_fit_range(wavelength)
    gauss_fit = multi_gaussian(wavelength[full_cut], pars, fit=False)
    hb_range = wavelength[full_cut] < 5500

    # We add back the continuum, scaled appropriately
    hb_cont = fit_df['hb_scale'].iloc[0] * \
        total_spec_df['cont_f_lambda'][full_cut][hb_range]
    gauss_fit[hb_range] = gauss_fit[hb_range] #+ hb_cont
    ha_cont = fit_df['ha_scale'].iloc[0] * \
        total_spec_df['cont_f_lambda'][full_cut][~hb_range]
    gauss_fit[~hb_range] = gauss_fit[~hb_range] #+ ha_cont

    # Plots the spectrum and fit on all axes
    for axis in axes_arr:
        axis.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
        if axis_group > -1:
            axis.plot(cont_sub_df['wavelength_cut'], cont_sub_df['continuum_sub_ydata'], color='mediumseagreen', label='Continuum-Subtracted', marker='o', ls='None')
            if np.median(fast_continuum)>0:
                axis.plot(wavelength, fast_continuum, color='blue', label='Scaled FAST Cont')
        axis.plot(wavelength[full_cut][hb_range], gauss_fit[hb_range], color='orange',
                  lw=1, label='Gaussian Fit')
        axis.plot(wavelength[full_cut][~hb_range], gauss_fit[~hb_range], color='orange',
                  lw=1)
        
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

    # Plots the continuum scaled the the right height
    # ax_Ha.plot(wavelength, continuum * fit_df['ha_continuum_offset'].iloc[0],
    #            color='red', lw=1, label=f'Continuum model (scaled)')
    # ax_Hb.plot(wavelength, continuum * fit_df['hb_continuum_offset'].iloc[0],
    #            color='red', lw=1, label=f'Continuum model (scaled)')

    Ha_plot_range = (6530, 6600)  # Angstrom
    Hb_plot_range = (4840, 5030)
    Hb_plot_range = (4845, 4875)
    # Hb_plot_range = (4995, 5015)

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

    # ax.set_ylim(-1 * 10**-20, 1.01 * np.max(spectrum))
    ax.set_ylim(np.percentile(spectrum, [1, 99]))

    ax.legend(loc=1, fontsize=axisfont - 3)

    ax.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    if axis_group > -1:
        fig.savefig(
            imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_images/{axis_group}_emission_fit.pdf')
    elif scaled == 'True':
        fig.savefig(imd.emission_fit_images_dir +
                    f'/{groupID}_emission_fit_scaled.pdf')
    elif run_name != 'False':
        imd.check_and_make_dir(imd.prospector_emission_fits_dir +
                    f'/{run_name}_emission_plots')
        fig.savefig(imd.prospector_emission_fits_dir +
                    f'/{run_name}_emission_plots/{groupID}_emission_fit.pdf')
    
    else:
        fig.savefig(imd.emission_fit_images_dir +
                    f'/{groupID}_emission_fit.pdf')
    plt.close()
    return


def norm_gausses(wavelength, peak_wavelength, sig):
    """Normalized Gaussians

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    peak_wavelength (float): Peak of the line in the rest frame [angstroms]
    sig (float): Standard deviation of the gaussian [angstroms]

    Returns:
    """
    A48, A63, A83, B = get_amps(wavelength, peak_wavelength, sig)
    gauss = np.exp(-0.5 * (wavelength - (peak_wavelength))**2 /
                   (np.e**sig)**2) / np.sqrt(2 * np.pi * (np.e**sig)**2)
    s = A48 * g48 + A63 * g63 + A83 * g83 + B * m
    return s


def get_amps(wavelength, peak_wavelength, sig, spectrum, model):
    """Finds the amplitudes for a range of Gaussians

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    peak_wavelength (float): Peak of the line in the rest frame [angstroms]
    sig (float): Standard deviation of the gaussian [angstroms]
    spectrum (): Spectrum
    model (): Continuum model

    Returns:
    amps_and_offset (list) = First n elemenst are the n amplitudes, last element is the continuum offset
    """
    gauss = np.exp(-0.5 * (wavelength - (peak_wavelength))**2 /
                   (np.e**sig)**2) / np.sqrt(2 * np.pi * (np.e**sig)**2)
    amps_and_offset = nnls(np.transpose([g48, g63, g83, model]), spectrum)[0]
    return amps_and_offset


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


def monte_carlo_fit(func, wavelength_cut, continuum, y_data, y_err, guess, bounds, n_loops, fit_axis_group=0, fast_continuum_cut=0):
    '''Fit the multi-gaussian to the data, use monte carlo to get uncertainties

    Parameters:
    wavelength_cut (pd.DataFrame): x data to be fit, 1d
    continuum (pd.DataFrame): Continuum to be scaled and subtracted off of the y_data
    y_data (pd.DataFrame): y data to be fit, 1d
    y_err (pd.DataFrame): Uncertainties on the y_data
    guess (list): list of guesses for the parameters of the fit
    bounds (tuple of array-like): bounds for the fit
    n_loops (int): Number of times to run the monte_carlo simulations
    fit_axis_group (int): Set to 1 if fitting an axis group, and add fast_coontinuum cut
    fast_continuum_cut (array): FAST conitnuum cut in the same way as y_data, only needed for fit_axis_group

    Returns:
    popt (list): List of the fit parameters
    err_popt (list): Uncertainties on these parameters
    '''
    # Ranges for ha and hb
    hb_half_idx = wavelength_cut < 5500
    ha_half_idx = np.logical_not(hb_half_idx)

    hb_cut = get_cuts(wavelength_cut[hb_half_idx])
    ha_cut = get_cuts(wavelength_cut[ha_half_idx])
    
    if fit_axis_group == 1:
        y_data_cont_sub, hb_scale, ha_scale = fast_continuum_subtract(y_data, fast_continuum_cut, hb_half_idx, ha_half_idx)
    else:
        y_data_cont_sub, hb_scale, ha_scale = scale_continuum(y_data, continuum, hb_half_idx, ha_half_idx, hb_cut, ha_cut)

    start = time.time()

    popt, pcov = curve_fit(func, wavelength_cut,
                           y_data_cont_sub, guess, bounds=bounds)
    end = time.time()
    print(f'Length of one fit: {end-start}')
    start = time.time()

    start = time.time()
    # Create a new set of y_datas
    new_y_datas = [[np.random.normal(loc=y_data.iloc[j], scale=y_err.iloc[
                                     j]) for j in range(len(y_data))] for i in range(n_loops)]
    # Turn them into dataframes with matching indicies
    new_y_data_dfs = [pd.DataFrame(new_y, columns=['flux']).set_index(
        y_data.index)['flux'] for new_y in new_y_datas]
    # Scale and subtract the continuum of of each
    
    if fit_axis_group == 1:
        new_cont_tuples = [fast_continuum_subtract(new_y_data_dfs[i], fast_continuum_cut, hb_half_idx, ha_half_idx) for i in range(len(new_y_data_dfs))]
    else:
        new_cont_tuples = [scale_continuum(new_y_data_dfs[i], continuum, hb_half_idx, ha_half_idx, hb_cut, ha_cut) for i in range(len(new_y_data_dfs))]

    
    new_y_datas_cont_sub = [cont_tuple[0] for cont_tuple in new_cont_tuples]
    hb_scales = [cont_tuple[1] for cont_tuple in new_cont_tuples]
    ha_scales = [cont_tuple[2] for cont_tuple in new_cont_tuples]

    fits_out = [curve_fit(func, wavelength_cut, new_y, guess, bounds=bounds)
                for new_y in new_y_datas_cont_sub]
    new_popts = [i[0] for i in fits_out]

    cont_scale_out = (hb_scale, ha_scale, np.std(hb_scales), np.std(ha_scales))

    end = time.time()
    print(f'Length of {n_loops} fits: {end-start}')
    return popt, new_popts, cont_scale_out, y_data_cont_sub


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


def get_flux(amp, sig, amp_err=0, sig_err=0):
    '''Given the amplitude and std deviation of a Gaussian, compute the line flux

    Parameters:
    amp (float): amplitude of gaussian (flux units)
    sig (float): Standard deviation of the gaussian (angstrom)

    Returns:
    flux (float): Total area under the Gaussian
    '''
    flux = amp * sig * np.sqrt(2 * np.pi)
    if amp_err != 0:
        amp_err_pct = amp_err / amp
        sig_err_pct = sig_err / sig
        flux_err_pct = amp_err_pct + sig_err_pct
        flux_err = flux * flux_err_pct
        return (flux, flux_err)
    return (flux, 0)


def get_flux_integrate(gaussian, continuum, wavelength_range):
    '''Given the amplitude and std deviation of a Gaussian, compute the line flux

    Parameters:
    gaussian (pd.DataFrame): y-values of the gaussian function
    continuum (pd.DataFrame): y-values of the continuum
    wavelength_range (pd.DataFrame): x-values to integrate along

    Returns:
    flux (float): Total area under the Gaussian
    '''
    integral_gauss = integrate.quad()
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


def fit_all_axis_ratio_emission(n_groups, save_name=''):
    """Runs the fit_emission() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters
    norm_method (str): Method of normalization

    Returns:
    """
    for i in range(n_groups):
        print(f'Fitting emission for axis ratio group {i}')
        fit_emission(i, 'cluster_norm', axis_group=i, save_name=save_name)


def get_fit_range(wavelength):
    """Gets the arrray of booleans that contains the two ranges to perform fitting

    Parameters:
    wavelength (pd.DataFrame): Dataframe of wavelength

    Returns:
    """
    cut_ha = np.logical_and(
        wavelength > 6500, wavelength < 6650)
    cut_hb = np.logical_and(
        wavelength > 4800, wavelength < 5050)
    full_cut = np.logical_or(cut_hb, cut_ha)
    return full_cut


def multi_gaussian(wavelength_cut, *pars, fit=True):
    """Fits all Gaussians simulatneously at fixed redshift

    Parameters:
    wavelength_cut (pd.DataFrame): Wavelength array to fit, just the two emission line regions concatenated together
    pars (list): List of all of the parameters
    fit (boolean): Set to True if fitting (ie amps are not constrained yet)

    Returns:
    combined_gauss (array): Gaussian function over the h_beta and h_alpha region concatenated
    """
    if len(pars) == 1:
        pars = pars[0]
    z_offset = pars[0]
    velocity = pars[1]

    # Split the wavelength into its Halpha nad Hbeta parts
    wavelength_hb = wavelength_cut[wavelength_cut < 5500]
    wavelength_ha = wavelength_cut[wavelength_cut > 5500]

    line_names = [line_list[i][0] for i in range(len(line_list))]

    hb_idxs = [i for i, line in enumerate(line_list) if line[0] in [
        'Hbeta', 'O3_5008', 'O3_4960']]
    ha_idxs = [i for i, line in enumerate(line_list) if line[0] not in [
        'Hbeta', 'O3_5008', 'O3_4960']]
    #start_2 = time.time()
    gaussians_hb = [gaussian_func(wavelength_hb, line_list[i][
                                  1] + z_offset, pars[i + 2], velocity_to_sig(line_list[i][1], velocity)) for i in hb_idxs]
    gaussians_ha = [gaussian_func(wavelength_ha, line_list[i][
                                  1] + z_offset, pars[i + 2], velocity_to_sig(line_list[i][1], velocity)) for i in ha_idxs]

    hb_y_vals = np.sum(gaussians_hb, axis=0)
    ha_y_vals = np.sum(gaussians_ha, axis=0)
    combined_gauss = np.concatenate([hb_y_vals, ha_y_vals])

    return combined_gauss


def scale_continuum(y_data_cut, continuum_cut, hb_half_idx, ha_half_idx, hb_cut, ha_cut):
    """Scales the continuum around the h_alpha and h_beta regions independently,k the n returns a subtracted version

    Parameters:
    y_data (array): flux/spectrum data, cut to only the h_alpha and h_beta regions
    continuum (array): continuum values, cut similarly
    hb_half_idx (array of booleans): idx of y_data and continuum_cut that correspond to h_beta. Opposite is ha_range
    ha_half_idx (array): See above
    hb_cut (array of booleans): After slicing by hb_half_idx, these are the booleans with emission lines removed
    ha_cut (array): See above

    Returns:
    y_data_cont_sub (array): Continuum subtracted y_data, only in the regions around h_alpha and h_beta
    """
    hb_cont = continuum_cut[hb_half_idx][hb_cut]
    hb_data = y_data_cut[hb_half_idx][hb_cut]
    hb_scale = np.sum(hb_data * hb_cont) / np.sum(hb_cont**2)

    ha_cont = continuum_cut[ha_half_idx][ha_cut]
    ha_data = y_data_cut[ha_half_idx][ha_cut]
    ha_scale = np.sum(ha_data * ha_cont) / np.sum(ha_cont**2)

    y_data_cut[hb_half_idx] = y_data_cut[hb_half_idx] - \
        continuum_cut[hb_half_idx] * hb_scale
    y_data_cut[ha_half_idx] = y_data_cut[ha_half_idx] - \
        continuum_cut[ha_half_idx] * ha_scale

    return (y_data_cut, hb_scale, ha_scale)

def fast_continuum_subtract(y_data_cut, fast_continuum_cut, hb_half_idx, ha_half_idx):
    """Uses the FAST continuum as the basis for subtraction, then returns the continuum subtacted data in just the ha and hb regions

    Parameters:
    y_data (array): flux/spectrum data, cut to only the h_alpha and h_beta regions
    continuum (array): continuum values, cut similarly
    hb_half_idx (array of booleans): idx of y_data and continuum_cut that correspond to h_beta. Opposite is ha_range
    ha_half_idx (array): See above

    Returns:
    y_data_cont_sub (array): Continuum subtracted y_data, only in the regions around h_alpha and h_beta
    """
    hb_scale = 1
    ha_scale = 1
    y_data_cut[hb_half_idx] = y_data_cut[hb_half_idx] - \
        fast_continuum_cut[hb_half_idx] * hb_scale
    y_data_cut[ha_half_idx] = y_data_cut[ha_half_idx] - \
        fast_continuum_cut[ha_half_idx] * ha_scale

    return (y_data_cut, hb_scale, ha_scale)


def get_cuts(wavelength_cut_section, width=7):
    """
    Parameters:
    wavelength_cut_section (array): wavelength data, cut to only the h_alpha OR h_beta region
    width (int): How many angstroms around the line to cut

    Returns:
    cut (array): mask for the emisison lines in the region
    """
    # Masks out the lines, cut is true everywhere else
    cuts = []
    for line_center in line_centers_rest:
        above = (wavelength_cut_section > line_center + width)
        below = (wavelength_cut_section < line_center - width)
        cuts.append(np.logical_or(above, below))
    cuts = np.prod(cuts, axis=0)
    cut = [bool(i) for i in cuts]
    return cut
