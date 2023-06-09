# Codes for simultaneously fitting the emission lines in a spectrum

from ensurepip import bootstrap
import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from sklearn import cluster
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
from axis_group_metallicities import compute_err_and_logerr, compute_O3N2_metallicity
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


def fit_emission(groupID, norm_method, constrain_O3=False, axis_group=-1, save_name='', scaled='False', run_name='False', bootstrap_num=-1):
    """Given a groupID, fit the emission lines in that composite spectrum

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    save_name(str): Folder of where to save and where spectra are located.
    constrain_O3 (boolean): Set to True to constrain the fitting of O3 to have a flux ratio of 2.97
    axis_group (int): Set to the number of the axis ratio group to fit that instead
    scaled (str): Set to 'True' if fitting the scaled spectra
    run_name (str): Set to the prospector run_name if fitting prospector spectra
    bootstrap_num (int): Set to -1 to avoid bootstrap, set to the number to read in the corresponding spectrum and fit that

    Returns:
    Saves a csv of the fits for all of the lines
    """
    # Number of loops in Monte Carlo
    n_loops = 0

    if axis_group > -1:
        if bootstrap_num > -1:
            composite_spectrum_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra_boots/{axis_group}_spectrum_{bootstrap_num}.csv').to_pandas()
            fast_continuum_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts_boots/{axis_group}_summed_cont_{bootstrap_num}.csv').to_pandas()
            fast_continuum = fast_continuum_df['f_lambda_scaled']
        else:
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
        composite_spectrum_df = read_composite_spectrum(groupID, norm_method, bootstrap_num=bootstrap_num)
        

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
        popt, arr_popt, cont_scale_out, y_data_cont_sub = monte_carlo_fit(multi_gaussian, wavelength_cut, scale_factor * continuum_cut, scale_factor * composite_spectrum_df[full_cut]['f_lambda'], scale_factor * composite_spectrum_df[full_cut]['err_f_lambda'], np.array(guess), bounds, n_loops)
    
    err_popt = np.std(arr_popt, axis=0)
    # If we're not doing the monte carlo fitting, just return -99s for all the uncertainties. These can be updated laters
    if n_loops == 0:
        err_popt = np.ones(len(popt))*-99

    # popt, pcov = curve_fit(multi_gaussian, composite_spectrum_df[
    #     'wavelength'], composite_spectrum_df['f_lambda'], guess)

    # Save the continuum-subtracted ydata
    if axis_group > -1:
        y_data_cont_sub = y_data_cont_sub / scale_factor
        cont_sub_df = pd.DataFrame(zip(wavelength_cut, y_data_cont_sub), columns=['wavelength_cut','continuum_sub_ydata'])
        if bootstrap_num > -1:
            imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs_boots/')
            cont_sub_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs_boots/{axis_group}_cont_sub_{bootstrap_num}.csv', index=False)
        else:
            cont_sub_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs/{axis_group}_cont_sub.csv', index=False)
    else:
        imd.check_and_make_dir(imd.emission_fit_dir + '/emission_fit_cont_subs/')
        cont_sub_df = pd.DataFrame(zip(wavelength_cut, y_data_cont_sub / scale_factor), columns=['wavelength_cut','continuum_sub_ydata'])
        cont_sub_df.to_csv(imd.emission_fit_dir + f'/emission_fit_cont_subs/{groupID}_cont_sub.csv', index=False)

    # Now, parse the results into a dataframe
    hb_scale, ha_scale, err_hb_scale, err_ha_scale = cont_scale_out
    hb_scales = [hb_scale for i in range(len(line_list))]
    ha_scales = [ha_scale for i in range(len(line_list))]
    err_hb_scales = [err_hb_scale for i in range(len(line_list))]
    err_ha_scales = [err_ha_scale for i in range(len(line_list))]
    line_names = [line_list[i][0] for i in range(len(line_list))]
    line_centers_rest = [line_list[i][1] for i in range(len(line_list))]
    z_offset = [popt[0] for i in range(len(line_list))]
    print(err_popt)
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
    oiii_idx = [idx for idx, name in enumerate(line_names) if name == 'O3_5008'][0]
    nii_idx = [idx for idx, name in enumerate(line_names) if name == 'N2_6585'][0]
    ha_amps = [arr_popt[i][2 + ha_idx] for i in range(len(arr_popt))]
    hb_amps = [arr_popt[i][2 + hb_idx] for i in range(len(arr_popt))]
    oiii_amps = [arr_popt[i][2 + oiii_idx] for i in range(len(arr_popt))]
    nii_amps = [arr_popt[i][2 + nii_idx] for i in range(len(arr_popt))]
    ha_sigs = [velocity_to_sig(line_list[ha_idx][1], arr_popt[i][
                               1])for i in range(len(arr_popt))]
    hb_sigs = [velocity_to_sig(line_list[hb_idx][1], arr_popt[i][
                               1])for i in range(len(arr_popt))]
    oiii_sigs = [velocity_to_sig(line_list[oiii_idx][1], arr_popt[i][
                               1])for i in range(len(arr_popt))]
    nii_sigs = [velocity_to_sig(line_list[nii_idx][1], arr_popt[i][
                               1])for i in range(len(arr_popt))]
    all_ha_fluxes = [get_flux(ha_amps[i]/scale_factor, ha_sigs[i])
                     for i in range(len(arr_popt))]
    all_hb_fluxes = [get_flux(hb_amps[i]/scale_factor, hb_sigs[i])
                     for i in range(len(arr_popt))]
    all_oiii_fluxes = [get_flux(oiii_amps[i]/scale_factor, oiii_sigs[i])
                     for i in range(len(arr_popt))]
    all_nii_fluxes = [get_flux(nii_amps[i]/scale_factor, nii_sigs[i]) for i in range(len(arr_popt))]
    all_balmer_decs = [all_ha_fluxes[i][0] / all_hb_fluxes[i][0]
                       for i in range(len(arr_popt))]
    balmer_dec = [fluxes[ha_idx] / fluxes[hb_idx]
                  for i in range(len(line_list))]

    # Compute metallicities
    N2_Ha_measures, log_N2_Ha_measures, _, _ = compute_err_and_logerr(fluxes[nii_idx], fluxes[ha_idx], -99, -99)
    O3N2_numeratos, log_O3_Hb_measures, _, _ = compute_err_and_logerr(fluxes[oiii_idx], fluxes[hb_idx], -99, -99) 
    O3N2_measures, log_O3N2_measures, _, _ = compute_err_and_logerr(O3N2_numeratos, N2_Ha_measures, -99, -99) 
    O3N2_metal, _ = compute_O3N2_metallicity(log_O3N2_measures, -99) 
    measured_log_N2_Ha = [log_N2_Ha_measures for i in range(len(line_list))]
    measured_log_O3_Hb = [log_O3_Hb_measures for i in range(len(line_list))]
    measured_O3N2_metal = [O3N2_metal for i in range(len(line_list))]
    #Metal errors
    O3N2_metals = []
    log_N2_Ha_measures_list = []
    log_O3_Hb_measures_list = []
    for i in range(len(arr_popt)):
        N2_Ha_measures, log_N2_Ha_measures, _, _ = compute_err_and_logerr(all_nii_fluxes[i][0], all_ha_fluxes[i][0], -99, -99)
        O3N2_numeratos, log_O3_Hb_measures, _, _ = compute_err_and_logerr(all_oiii_fluxes[i][0], all_hb_fluxes[i][0], -99, -99) 
        O3N2_measures, log_O3N2_measures, _, _ = compute_err_and_logerr(O3N2_numeratos, N2_Ha_measures, -99, -99) 
        O3N2_metal, _ = compute_O3N2_metallicity(log_O3N2_measures, -99) 
        O3N2_metals.append(O3N2_metal)
        log_N2_Ha_measures_list.append(log_N2_Ha_measures)
        log_O3_Hb_measures_list.append(log_O3_Hb_measures)
    if n_loops != 0:
        err_log_N2_Ha_measures_low, err_log_N2_Ha_measures_high = np.percentile(log_N2_Ha_measures_list, [16, 84])
        err_log_O3_Hb_measures_low, err_log_O3_Hb_measures_high = np.percentile(log_O3_Hb_measures_list, [16, 84])
        err_metal_low, err_metal_high = np.percentile(O3N2_metals, [16, 84])
        err_log_N2_Ha_measures_low_value = measured_log_N2_Ha[0] - err_log_N2_Ha_measures_low
        err_log_N2_Ha_measures_high_value = err_log_N2_Ha_measures_high - measured_log_N2_Ha[0]
        err_log_O3_Hb_measures_low_value = measured_log_O3_Hb[0] - err_log_O3_Hb_measures_low
        err_log_O3_Hb_measures_high_value = err_log_O3_Hb_measures_high - measured_log_O3_Hb[0]
        err_metal_low_value = O3N2_metal - err_metal_low
        err_metal_high_value = err_metal_high - O3N2_metal
        err_metal_low = [err_metal_low_value for i in range(len(line_list))]
        err_metal_high = [err_metal_high_value for i in range(len(line_list))]
        err_log_N2_Ha_measures_low = [err_log_N2_Ha_measures_low_value for i in range(len(line_list))]
        err_log_N2_Ha_measures_high = [err_log_N2_Ha_measures_high_value for i in range(len(line_list))]
        err_log_O3_Hb_measures_low = [err_log_O3_Hb_measures_low_value for i in range(len(line_list))]
        err_log_O3_Hb_measures_high = [err_log_O3_Hb_measures_high_value for i in range(len(line_list))]
    else:
        err_metal_low = -99*np.ones(len(balmer_dec))
        err_metal_high = -99*np.ones(len(balmer_dec))
        err_log_N2_Ha_measures_low = -99*np.ones(len(balmer_dec))
        err_log_N2_Ha_measures_high = -99*np.ones(len(balmer_dec))
        err_log_O3_Hb_measures_low = -99*np.ones(len(balmer_dec))
        err_log_O3_Hb_measures_high = -99*np.ones(len(balmer_dec))

    if n_loops == 0:
        err_balmer_dec_low = -99*np.ones(len(balmer_dec))
        err_balmer_dec_high = -99*np.ones(len(balmer_dec))
        err_ha_scales = -99*np.ones(len(balmer_dec))
        err_hb_scales = -99*np.ones(len(balmer_dec))
        err_sigs = -99*np.ones(len(balmer_dec))
        err_amps = -99*np.ones(len(balmer_dec))
        err_fluxes = -99*np.ones(len(balmer_dec))
    else:
        err_balmer_dec_low = balmer_dec - np.percentile(all_balmer_decs, 16)
        err_balmer_dec_high = np.percentile(all_balmer_decs, 84) - balmer_dec

    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, err_z_offset, hb_scales, err_hb_scales, ha_scales, err_ha_scales, velocity, err_velocity, amps, err_amps, sigs, err_sigs, fluxes, err_fluxes, balmer_dec, err_balmer_dec_low, err_balmer_dec_high, measured_O3N2_metal, err_metal_low, err_metal_high, measured_log_N2_Ha, err_log_N2_Ha_measures_low, err_log_N2_Ha_measures_high, measured_log_O3_Hb, err_log_O3_Hb_measures_low, err_log_O3_Hb_measures_high), columns=['line_name', 'line_center_rest', 'z_offset', 'err_z_offset', 'hb_scale', 'err_hb_scale', 'ha_scale', 'err_ha_scale', 'fixed_velocity', 'err_fixed_velocity', 'amplitude', 'err_amplitude', 'sigma', 'err_sigma', 'flux', 'err_flux', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high', 'O3N2_metallicity', 'err_O3N2_metallicity_low', 'err_O3N2_metallicity_high', 'log_N2_Ha', 'err_log_N2_Ha_low', 'err_log_N2_Ha_high', 'log_O3_Hb', 'err_log_O3_Hb_low', 'err_log_O3_Hb_high'])
    fit_df['signal_noise_ratio'] = fit_df['flux']/fit_df['err_flux']


    if axis_group > -1:
        if bootstrap_num > -1:
            imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits_boots/')
            fit_df.to_csv(
                imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits_boots/{axis_group}_emission_fits_{bootstrap_num}.csv', index=False)
            plot_emission_fit(groupID, norm_method,
                            axis_group=axis_group, save_name=save_name, bootstrap_num=bootstrap_num)
        else:
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
        # plot_emission_fit(groupID, norm_method, run_name=run_name)
    else:
        if bootstrap_num > -1:
            imd.check_and_make_dir(imd.emission_fit_csvs_dir)
            imd.check_and_make_dir(imd.emission_fit_images_dir)
            imd.check_and_make_dir(imd.emission_fit_dir +
                        f'/emission_fitting_boot_csvs/')
            fit_df.to_csv(imd.emission_fit_dir +
                        f'/emission_fitting_boot_csvs/{groupID}_emission_fits_{bootstrap_num}.csv', index=False)
        else:
            imd.check_and_make_dir(imd.emission_fit_csvs_dir)
            imd.check_and_make_dir(imd.emission_fit_images_dir)
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


def plot_emission_fit(groupID, norm_method, axis_group=-1, save_name='', scaled='False', run_name='False', bootstrap_num=-1):
    """Plots the fit to each emission line

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    axis_group (int): Set to the number of the axis ratio group to fit that instead
    scaled (str): Set to true if plotting the scaled fits
    run_name (str): Set to name of prospector run to fit with those
    bootstrap_num (int): Which number in the bootstrap to plot, -1 to plot the original

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
        if bootstrap_num > -1:
            fit_df = ascii.read(
                imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits_boots/{axis_group}_emission_fits_{bootstrap_num}.csv').to_pandas()
            total_spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra_boots/{axis_group}_spectrum_{bootstrap_num}.csv').to_pandas()
            fast_continuum_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts_boots/{axis_group}_summed_cont_{bootstrap_num}.csv').to_pandas()
            fast_continuum = fast_continuum_df['f_lambda_scaled']
            cont_sub_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs_boots/{axis_group}_cont_sub_{bootstrap_num}.csv').to_pandas()
        else:
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
        # cont_sub_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_cont_sub.csv').to_pandas()
    else:
        fit_df = ascii.read(imd.emission_fit_csvs_dir +
                            f'/{groupID}_emission_fits.csv').to_pandas()
        total_spec_df = read_composite_spectrum(groupID, norm_method)
        cont_sub_df = ascii.read(imd.emission_fit_dir + f'/emission_fit_cont_subs/{groupID}_cont_sub.csv').to_pandas()

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
    gauss_fit_df = pd.DataFrame(zip(wavelength[full_cut], gauss_fit), columns=['rest_wavelength', 'gaussian_fit'])
    imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_gaussian_fits/')
    gauss_fit_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_gaussian_fits/{axis_group}_gaussian_fit.csv', index=False)
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
        axis.plot(cont_sub_df['wavelength_cut'], cont_sub_df['continuum_sub_ydata'], color='mediumseagreen', label='Continuum-Subtracted', marker='o', ls='None')
        if axis_group > -1:
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
            wavelength > plot_range[0], wavelength < plot_range[1])]) - 2e-19
        lim_max = 1.05 * np.max(spectrum[np.logical_and(
            wavelength > plot_range[0], wavelength < plot_range[1])])
        axis.set_ylim(lim_min, lim_max)
        axis.set_xlim(plot_range)
        rect = patches.Rectangle((plot_range[0], lim_min), (plot_range[
                                 1] - plot_range[0]), (lim_max - lim_min), linewidth=1.5, edgecolor=box_color, facecolor='None')
        ax.add_patch(rect)

    set_plot_ranges(ax, ax_Ha, Ha_plot_range, Ha_zoom_box_color)
    set_plot_ranges(ax, ax_Hb, Hb_plot_range, Hb_zoom_box_color)

    ax_Ha.text(0.05, 0.93, f"Ha: {round(10**17*fit_df.iloc[0]['flux'], 4)}", transform=ax_Ha.transAxes)
    ax_Hb.text(0.05, 0.93, f"Hb: {round(10**17*fit_df.iloc[1]['flux'], 4)}", transform=ax_Hb.transAxes)
    ax_Hb.text(0.05, 0.83, f"BalmDec: {round(fit_df.iloc[1]['balmer_dec'], 4)}", transform=ax_Hb.transAxes)

    # ax.set_ylim(-1 * 10**-20, 1.01 * np.max(spectrum))
    ax.set_ylim(np.percentile(spectrum, [1, 99]))

    ax.legend(loc=1, fontsize=axisfont - 3)

    ax.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    if axis_group > -1:
        if bootstrap_num > -1:
            imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_images_boots/')
            fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_images_boots/{axis_group}_emission_fit_{bootstrap_num}.pdf')
        else:
            fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_images/{axis_group}_emission_fit.pdf')
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

    y_data = y_data.fillna(0)
    y_data_cont_sub = y_data_cont_sub.fillna(0)

    start = time.time()

    popt, pcov = curve_fit(func, wavelength_cut,
                           y_data_cont_sub, guess, bounds=bounds)
    end = time.time()
    print(f'Length of one fit: {end-start}')
    start = time.time()

    start = time.time()
    # Create a new set of y_datas

    #Fill over nan values with the median error * 3
    y_err = y_err.fillna(5*y_err.median())

    # np.random.normal(loc=y_data.iloc[627], scale=y_err.iloc[627])
    # [np.random.normal(loc=y_data.iloc[jsq], scale=y_err.iloc[jsq]) for jsq in range(798)]
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

def get_amp(flux, sig):
    '''Given the flux and std deviation of a Gaussian, compute the amplitude

    Parameters:
    flux (float): flux of gaussian (flux units)
    sig (float): Standard deviation of the gaussian (angstrom)

    Returns:
    flux (float): Total area under the Gaussian
    '''
    amp = flux / (sig * np.sqrt(2 * np.pi))
    return amp



def fit_all_emission(n_clusters, norm_method, ignore_groups, constrain_O3=False, bootstrap=-1):
    """Runs the fit_emission() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters
    norm_method (str): Method of normalization

    Returns:
    """
    for i in range(n_clusters):
        if i in ignore_groups:
            print(f'Ignoring group {i}')
            continue
        print(f'Fitting emission for {i}')
        fit_emission(i, norm_method, constrain_O3=constrain_O3)
        if bootstrap > -1:
            for bootstrap_num in range(bootstrap):
                fit_emission(i, norm_method, constrain_O3=constrain_O3, bootstrap_num=bootstrap_num)


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

    y_data_copy = y_data_cut.copy()

    y_data_copy[hb_half_idx] = y_data_copy[hb_half_idx] - \
        fast_continuum_cut[hb_half_idx] * hb_scale
    y_data_copy[ha_half_idx] = y_data_copy[ha_half_idx] - \
        fast_continuum_cut[ha_half_idx] * ha_scale

    return (y_data_copy, hb_scale, ha_scale)


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

def compute_bootstrap_uncertainties(n_clusters, save_name, bootstrap=-1, clustering=False, ignore_groups=[], ha_first=False):
    """Reads in all the bootstrapped fits form all the clusters, then computes uncertainties and adds them back to the main fit
    
    Parameters:
    n_clusters (int): Number of axis ratio groups
    save_name (str): Name of the folder they are saved in
    bootstrap (int): Set to the number of bootstrapped data points
    clustering (boolean): Set to true if using clusters, will grab files from cluster_dir
    ignore_groups (list): Fill with groupIDs to skip 
    """
    for axis_group in range(n_clusters):
        if axis_group in ignore_groups:
            print(f'Ignoring group {axis_group}')
            continue
        if clustering == True:
            groupID = axis_group
            if ha_first == True:
                emission_df_loc = imd.emission_fit_dir + '/ha_first_csvs' + f'/{groupID}_emission_fits.csv'
                boot_dfs = [ascii.read(imd.emission_fit_dir + f'/ha_first_boot_csvs/{groupID}_emission_fits_{bootstrap_num}.csv').to_pandas() for bootstrap_num in range(bootstrap)]
            else:
                emission_df_loc = imd.emission_fit_csvs_dir + f'/{groupID}_emission_fits.csv'
                boot_dfs = [ascii.read(imd.emission_fit_dir + f'/emission_fitting_boot_csvs/{groupID}_emission_fits_{bootstrap_num}.csv').to_pandas() for bootstrap_num in range(bootstrap)]
        else:
            emission_df_loc = imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv'

            boot_dfs = [ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits_boots/{axis_group}_emission_fits_{bootstrap_num}.csv').to_pandas() for bootstrap_num in range(bootstrap)]
        emission_df = ascii.read(emission_df_loc).to_pandas()

        def compute_err_on_col(col_name, true_values, symmetric_err=True):
            """Give a column name, use the bootstrapped points to find an uncertainty
            
            Parameters:
            col_name(str): Column name that you are comping the uncertainties for, e.g. 'flux'
            true_values (pd.DataFrame of float): Value measured in the emission_df
            symmetric_err (boolean): Set to True to just get a standard deviation, flase for asymmetric errors
            """
            arr = np.array([boot_dfs[i][col_name] for i in range(len(boot_dfs))])
            if symmetric_err==True:
                errs = np.std(arr, axis=0)
                low_errs = -99*np.ones(len(arr))
                high_errs = -99*np.ones(len(arr))
            else:
                errs = -99*np.ones(len(arr))
                low_errs = true_values - np.percentile(arr, 16, axis=0)
                high_errs = np.percentile(arr, 84, axis=0) - true_values
            return errs, low_errs, high_errs

        err, low_err, high_err = compute_err_on_col('z_offset', emission_df['z_offset'], symmetric_err=True)
        emission_df['err_z_offset'] = err

        err, low_err, high_err = compute_err_on_col('hb_scale', emission_df['hb_scale'], symmetric_err=True)
        emission_df['err_hb_scale'] = err

        err, low_err, high_err = compute_err_on_col('ha_scale', emission_df['ha_scale'], symmetric_err=True)
        emission_df['err_ha_scale'] = err

        err, low_err, high_err = compute_err_on_col('fixed_velocity', emission_df['fixed_velocity'], symmetric_err=True)
        emission_df['err_fixed_velocity'] = err

        err, low_err, high_err = compute_err_on_col('amplitude', emission_df['amplitude'], symmetric_err=True)
        emission_df['err_amplitude'] = err

        err, low_err, high_err = compute_err_on_col('sigma', emission_df['sigma'], symmetric_err=True)
        emission_df['err_sigma'] = err

        err, low_err, high_err = compute_err_on_col('flux', emission_df['flux'], symmetric_err=True)
        emission_df['err_flux'] = err

        err, low_err, high_err = compute_err_on_col('balmer_dec', emission_df['balmer_dec'], symmetric_err=False)
        emission_df['err_balmer_dec_low'] = low_err
        emission_df['err_balmer_dec_high'] = high_err

        # Ha_first is currently not fitting O3 lines or metallicities
        if ha_first == True:
            pass
        else:
            err, low_err, high_err = compute_err_on_col('O3N2_metallicity', emission_df['O3N2_metallicity'], symmetric_err=False)
            emission_df['err_O3N2_metallicity_low'] = low_err
            emission_df['err_O3N2_metallicity_high'] = high_err

            err, low_err, high_err = compute_err_on_col('log_N2_Ha', emission_df['log_N2_Ha'], symmetric_err=False)
            emission_df['err_log_N2_Ha_low'] = low_err
            emission_df['err_log_N2_Ha_high'] = high_err

            err, low_err, high_err = compute_err_on_col('log_O3_Hb', emission_df['log_O3_Hb'], symmetric_err=False)
            emission_df['err_log_O3_Hb_low'] = low_err
            emission_df['err_log_O3_Hb_high'] = high_err

        

        emission_df['signal_noise_ratio'] = emission_df['flux']/emission_df['err_flux']
        emission_df.to_csv(emission_df_loc, index=False)
# fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=0, save_name='both_ssfrs_4bin_median_2axis', scaled='False', run_name='False', bootstrap_num=0)
# bootstrap = 10
# fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=1, save_name='both_ssfrs_4bin_median_2axis', scaled='False', run_name='False')
# for bootstrap_num in range(bootstrap):
#     fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=1, save_name='both_ssfrs_4bin_median_2axis', scaled='False', run_name='False', bootstrap_num=bootstrap_num)
# compute_bootstrap_uncertainties(8, 'both_ssfrs_4bin_median_2axis', bootstrap=10)
# for axis_group in range(8):
#     plot_emission_fit(0, 'cluster_norm', axis_group=axis_group, save_name='both_sfms_4bin_median_2axis_boot100')

# fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=0, save_name='norm_1_sn3_filtered_cont_renorm', scaled='False', run_name='False')