# Codes for simultaneously fitting the emission lines in a spectrum
### MODIFIED VERSION OF fit_emission.py, look there for most up-to-date


from ensurepip import bootstrap
import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed, read_fast_continuum
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from scipy.optimize import curve_fit, minimize, leastsq
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from spectra_funcs import read_axis_ratio_spectrum, read_composite_spectrum, get_too_low_gals, get_spectra_files, read_spectrum
import matplotlib.patches as patches
import time
from axis_ratio_funcs import read_filtered_ar_df


line_list = [
    ('Halpha', 6564.61)
]
line_centers_rest = [line_list[i][1] for i in range(len(line_list))]


def fit_emission(spectrum_df, fast_continuum_df, save_name='', scaled='False', run_name='False', bootstrap_num=-1):
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
    n_loops = 100

    line_names = [line_list[i][0] for i in range(len(line_list))]

    # Scale factor to make fitting more robust
    scale_factor = 10**18

    # Build the initial guesses
    guess = []
    bounds_low = []
    bounds_high = []
    amp_guess = 10**-18  # flux units
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

    wavelength = spectrum_df['rest_wavelength']
    interp_fast_continuum_df = interp_cont_to_spec(spectrum_df, fast_continuum_df)
    full_cut = get_fit_range(wavelength)
    wavelength_cut = wavelength[full_cut]
    fast_continuum = interp_fast_continuum_df['f_lambda']
    fast_continuum_cut = fast_continuum[full_cut]
    # print(bounds)
    # print(guess)
    
    popt, arr_popt, cont_scale_out, y_data_cont_sub = monte_carlo_fit(multi_gaussian, wavelength_cut, scale_factor * fast_continuum_cut, scale_factor * spectrum_df[full_cut][
        'f_lambda'], scale_factor * spectrum_df[full_cut]['err_f_lambda'], np.array(guess), bounds, n_loops)
    
    err_popt = np.std(arr_popt, axis=0)
    # If we're not doing the monte carlo fitting, just return -99s for all the uncertainties. These can be updated laters
    if n_loops == 0:
        err_popt = np.ones(len(popt))*-99

    # popt, pcov = curve_fit(multi_gaussian, composite_spectrum_df[
    #     'wavelength'], composite_spectrum_df['f_lambda'], guess)

    # Save the continuum-subtracted ydata
    # y_data_cont_sub = y_data_cont_sub / scale_factor
    # cont_sub_df = pd.DataFrame(zip(wavelength_cut, y_data_cont_sub), columns=['wavelength_cut','continuum_sub_ydata'])
    # if bootstrap_num > -1:
    #     imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs_boots/')
    #     cont_sub_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs_boots/{axis_group}_cont_sub_{bootstrap_num}.csv', index=False)
    # else:
    #     cont_sub_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs/{axis_group}_cont_sub.csv', index=False)


    # Now, parse the results into a dataframe
    ha_scale, err_ha_scale = cont_scale_out
    ha_scales = [ha_scale for i in range(len(line_list))]
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
    # ha_amps = [arr_popt[i][2 + ha_idx] for i in range(len(arr_popt))]
    # ha_sigs = [velocity_to_sig(line_list[ha_idx][1], arr_popt[i][
    #                            1])for i in range(len(arr_popt))]
    # all_ha_fluxes = [get_flux(ha_amps[i], ha_sigs[i])
    #                  for i in range(len(arr_popt))]

    if n_loops == 0:
        err_ha_scales = [-99]
        err_amps = [-99]
        err_sigs = [-99]
        err_fluxes = [-99]
        #     err_ha_scales = -99*np.ones(len(all_ha_fluxes))
        #     err_sigs = -99*np.ones(len(all_ha_fluxes))
        #     err_amps = -99*np.ones(len(all_ha_fluxes))
        #     err_fluxes = -99*np.ones(len(all_ha_fluxes))
    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, err_z_offset, ha_scales, err_ha_scales, velocity, err_velocity, amps, err_amps, sigs, err_sigs, fluxes, err_fluxes), columns=['line_name', 'line_center_rest', 'z_offset', 'err_z_offset', 'ha_scale', 'err_ha_scale', 'fixed_velocity', 'err_fixed_velocity', 'amplitude', 'err_amplitude', 'sigma', 'err_sigma', 'flux', 'err_flux'])
    fit_df['signal_noise_ratio'] = fit_df['flux']/fit_df['err_flux']

    imd.check_and_make_dir(imd.emission_fit_indiv_dir)
    fit_df.to_csv(imd.emission_fit_indiv_dir + f'/{save_name}.csv', index=False)
    
    # Set up the parameters from the fitting
    pars = []
    pars.append(fit_df['z_offset'].iloc[0])
    pars.append(fit_df['fixed_velocity'].iloc[0])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['amplitude'])

    full_cut = get_fit_range(wavelength)
    gauss_fit = multi_gaussian(wavelength[full_cut], pars, fit=False)

    fig, ax = plt.subplots(figsize=(9,8))
    plot_ranges = [6540, 6590]
    spec_plot_vals = fast_continuum[full_cut]+y_data_cont_sub/scale_factor
    plot_idx = np.logical_and(wavelength_cut > plot_ranges[0], wavelength_cut<plot_ranges[1])
    ax.plot(wavelength_cut, spec_plot_vals, color='black', label='spectrum')
    ax.plot(wavelength_cut, fast_continuum[full_cut]+gauss_fit, color='orange', label='gaussian fit')
    ax.plot(wavelength_cut, fast_continuum[full_cut], color='blue', label='model continuum')
    ax.legend(fontsize=16)
    ax.set_xlim(plot_ranges)
    ax.set_ylim(np.min(spec_plot_vals[plot_idx])*1.05, np.max(spec_plot_vals[plot_idx])*1.05)
    ax.set_xlabel('Rest Wavelength', fontsize=18)
    ax.set_ylabel('Flux', fontsize=18)
    ax.text(0.05, 0.95, f'Flux: {round(fit_df["flux"].iloc[0]*10**17, 3)} e-17', transform=ax.transAxes, fontsize=16)
    ax.text(0.05, 0.89, f'S/N: {round(fit_df["signal_noise_ratio"].iloc[0], 3)}', transform=ax.transAxes, fontsize=16)
    ax.tick_params(labelsize=18)
    imd.check_and_make_dir(imd.emission_fit_indiv_dir_images)
    fig.savefig(imd.emission_fit_indiv_dir_images + f'/{save_name}.pdf')
    plt.close('all')
    return


def interp_cont_to_spec(spectrum_df, continuum_df):
    """ Interpolates the continuum to the spectrum
    """
    target_wavelengths = spectrum_df['rest_wavelength']
    interp_cont = interpolate.interp1d(
                    continuum_df['rest_wavelength'], continuum_df['f_lambda_rest'])
    interp_cont_values = interp_cont(target_wavelengths)
    interp_cont_df = pd.DataFrame(zip(target_wavelengths, interp_cont_values), columns=['rest_wavelength', 'f_lambda'])
    return interp_cont_df
    # fig, ax = plt.subplots(figsize=(9,8))
    # ha_range = np.logical_and(spectrum_df['rest_wavelength'] > 6540, spectrum_df['rest_wavelength'] < 6580)
    # ax.plot(target_wavelengths[ha_range], spectrum_df[ha_range]['f_lambda'], color='black', marker='o')
    # ax.plot(target_wavelengths[ha_range], interp_cont_values[ha_range], color='orange', marker='o')
    
    

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


def monte_carlo_fit(func, wavelength_cut, fast_continuum, y_data, y_err, guess, bounds, n_loops):
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
    y_data_cont_sub, ha_scale = fast_continuum_subtract(y_data, fast_continuum)
    
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
    
   
    new_cont_tuples = [fast_continuum_subtract(new_y_data_dfs[i], fast_continuum) for i in range(len(new_y_data_dfs))]
  
    
    new_y_datas_cont_sub = [cont_tuple[0] for cont_tuple in new_cont_tuples]
    ha_scales = [cont_tuple[1] for cont_tuple in new_cont_tuples]

    fits_out = [curve_fit(func, wavelength_cut, new_y, guess, bounds=bounds)
                for new_y in new_y_datas_cont_sub]
    new_popts = [i[0] for i in fits_out]

    cont_scale_out = (ha_scale, np.std(ha_scales))

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
    # For this, we don't need to measure hbeta region
    full_cut = cut_ha
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
    # Currently just a single gaussian that could fit Halpha region
    if len(pars) == 1:
        pars = pars[0]
    z_offset = pars[0]
    velocity = pars[1]

    # Split the wavelength into its Halpha nad Hbeta parts
    wavelength_ha = wavelength_cut[wavelength_cut > 5500]

    line_names = [line_list[i][0] for i in range(len(line_list))]

    ha_idxs = [i for i, line in enumerate(line_list) if line[0] not in [
        'Hbeta', 'O3_5008', 'O3_4960']]
    #start_2 = time.time()
    gaussians_ha = [gaussian_func(wavelength_ha, line_list[i][
                                  1] + z_offset, pars[i + 2], velocity_to_sig(line_list[i][1], velocity)) for i in ha_idxs]

    ha_y_vals = np.sum(gaussians_ha, axis=0)
    combined_gauss = ha_y_vals

    return combined_gauss


def scale_continuum(y_data_cut, continuum_cut):
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

    ha_cont = continuum_cut
    ha_data = y_data_cut
    ha_scale = np.sum(ha_data * ha_cont) / np.sum(ha_cont**2)

    y_data_cut = y_data_cut - continuum_cut * ha_scale

    return (y_data_cut, ha_scale)

def fast_continuum_subtract(y_data_cut, fast_continuum_cut):
    """Uses the FAST continuum as the basis for subtraction, then returns the continuum subtacted data in just the ha and hb regions

    Parameters:
    y_data (array): flux/spectrum data, cut to only the h_alpha and h_beta regions
    continuum (array): continuum values, cut similarly
    hb_half_idx (array of booleans): idx of y_data and continuum_cut that correspond to h_beta. Opposite is ha_range
    ha_half_idx (array): See above

    Returns:
    y_data_cont_sub (array): Continuum subtracted y_data, only in the regions around h_alpha and h_beta
    """
    ha_scale = 1
    y_data_cut = y_data_cut - fast_continuum_cut * ha_scale

    return (y_data_cut, ha_scale)


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


def fit_all_indiv_halpha():
    """Main method
    """
    # Read in the axis ratio dataframe
    ar_df = read_filtered_ar_df()

    for i in range(len(ar_df)):
        if i < 30:
            continue
        field = ar_df.iloc[i]['field']
        v4id = ar_df.iloc[i]['v4id']
        mosdef_obj = get_mosdef_obj(field, v4id)

        # Find all the spectra files corresponding to this object
        spectra_files = get_spectra_files(mosdef_obj)
        fast_continuum_df = read_fast_continuum(mosdef_obj)
        for spectrum_file in spectra_files:
            spectrum_df = read_spectrum(mosdef_obj, spectrum_file) # units already corrected by 1+z
            if spectrum_df.iloc[0]['rest_wavelength']<line_centers_rest[0]<spectrum_df.iloc[-1]['rest_wavelength']:
                try:
                    print(f'Fitting {field} {v4id}, number {i}')
                    fit_emission(spectrum_df, fast_continuum_df, save_name = f'{field}_{v4id}_halpha_fit')
                except RuntimeError:
                    print(f'Could not fit {field} {v4id}')
                    continue                # 
            else:
                continue

# fit_all_indiv_halpha()
