# Codes for simultaneously fitting the emission lines in a spectrum

import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import initialize_mosdef_dirs as imd
from axis_group_metallicities import compute_err_and_logerr, compute_O3N2_metallicity
import matplotlib.patches as patches
import time
from uncover_read_data import read_raw_spec, read_prism_lsf, read_fluxcal_spec, get_id_msa_list
from astropy.convolution import convolve
from scipy.interpolate import interp1d
from compute_av import compute_ha_pab_av, get_nii_correction
from plot_vals import scale_aspect

emission_fit_dir = '/Users/brianlorenz/uncover/Data/emission_fitting/'

line_list = [
    ('Halpha', 6564.6),
    ('PaBeta', 12821.7)
]
lines_dict = {
    line_list[0][0]: line_list[0][1],
    line_list[1][0]: line_list[1][1]
}
line_centers_rest = [line_list[i][1] for i in range(len(line_list))]

ha_fit_range = (5500, 7700)
pab_fit_range = (11800, 14200)


def fit_emission_uncover(spectrum, save_name, bootstrap_num=-1):
    """
    Parameters:
    bootstrap_num (int): Set to -1 to avoid bootstrap, set to the number to read in the corresponding spectrum and fit that
    
    Returns:
    Saves a csv of the fits for all of the lines
    """
    # Number of loops in Monte Carlo
    if bootstrap_num > -1:
        n_loops = 0
    else:
        n_loops = 10
    


    line_names = [line_list[i][0] for i in range(len(line_list))]

    scale_factor = 10**19

    guess_df = ascii.read('amp_guess_list.txt', header_start=0).to_pandas()
    guess_id_list = guess_df.id_msa.to_list()

    # Build the initial guesses
    guess = []
    bounds_low = []
    bounds_high = []
    amp_guess = 0.5  # flux units, take into accoutn scale factor
    velocity_guess = 1000  # km/s
    z_offset_guess = 0  # Angstrom
    # continuum_offset_guess = 10**-20  # flux untis,
    continuum_offset_guess = 0.5  # Scale to multiply by FAST continuum,
    # should eventually divide out the shape of continuum
    # First, we guess the z_offset
    for i in range(len(line_list)):
        guess.append(z_offset_guess)
        bounds_low.append(-10) # angstrom
        bounds_high.append(10)
    # Then, for each line, we guess a veolcity and amplitude
    for i in range(len(line_list)):
        guess.append(velocity_guess)
        bounds_low.append(0.01)
        bounds_high.append(100000)
    for i in range(len(line_list)):
        # if 'O3_5008' in line_names and 'O3_4960' in line_names:
        #     idx_5008 = line_names.index('O3_5008')
        #     idx_4960 = line_names.index('O3_4960')
        #     if i == idx_5008:
        #         guess.append(1)
        #         continue
        if i == 0:
            if save_name in guess_id_list:
                ha_guess = guess_df[guess_df['id_msa'] == save_name]['ha_amp'].iloc[0]
                guess.append(ha_guess) 
            else:
                guess.append(amp_guess) 
        if i == 1:
            if save_name in guess_id_list:
                pab_guess = guess_df[guess_df['id_msa'] == save_name]['pab_amp'].iloc[0]
                guess.append(pab_guess) 
            else:
                guess.append(amp_guess/8)
        bounds_low.append(0.001)
        bounds_high.append(10000000)
    bounds = (np.array(bounds_low), np.array(bounds_high))

    
    wavelength = spectrum['rest_wave_aa']
    flux = spectrum['rest_flux_calibrated_erg_aa']
    err_flux = spectrum['err_rest_flux_calibrated_erg_aa']
    # Set Drop the nans, and set wavelength to cover it
    flux = flux.dropna()
    wavelength = wavelength[flux.index]
    err_flux = err_flux[flux.index]
    full_cut = get_fit_range(wavelength)
    flux = flux[full_cut]
    err_flux = err_flux[full_cut]
    wavelength = wavelength[full_cut]
    if '_flat' in str(save_name):
        cont_flux = np.ones(len(wavelength)) * 1.5e-6 # From the blackbody code
        c = 299792458 # m/s
        continuum = cont_flux * (1e-23*1e10*c / (wavelength**2))
        continuum = pd.Series(continuum)
    else:
        continuum = fit_continuum(wavelength, flux, save_name=save_name)

    def get_cont_value_at_line(line_wave, continuum, wavelength):
        line_idx = np.argmin(np.abs(wavelength-line_wave))
        cont_value = continuum.iloc[line_idx]
        return cont_value
    ha_cont_value = get_cont_value_at_line(line_list[0][1], continuum, wavelength)
    pab_cont_value = get_cont_value_at_line(line_list[1][1], continuum, wavelength)
    cont_values = [ha_cont_value, pab_cont_value]

    popt, arr_popt, y_data_cont_sub = monte_carlo_fit(multi_gaussian, wavelength, scale_factor * continuum, scale_factor * flux, scale_factor * err_flux, np.array(guess), bounds, n_loops)
    err_popt = np.std(arr_popt, axis=0)
    
    
    # If we're not doing the monte carlo fitting, just return -99s for all the uncertainties. These can be updated laters
    if n_loops == 0:
        err_popt = np.ones(len(popt))*-99


    # Save the continuum-subtracted ydata
    imd.check_and_make_dir(emission_fit_dir)
    cont_sub_df = pd.DataFrame(zip(wavelength, y_data_cont_sub / scale_factor), columns=['wavelength','continuum_sub_ydata'])
    cont_sub_df.to_csv(f'{emission_fit_dir}{save_name}_cont_sub.csv', index=False)

    # Now, parse the results into a dataframe
    line_names = [line_list[i][0] for i in range(len(line_list))]
    line_centers_rest = [line_list[i][1] for i in range(len(line_list))]
    z_offset = [popt[0] for i in range(len(line_list))]
    print(err_popt)
    err_z_offset = [err_popt[0] for i in range(len(line_list))]
    velocities = [popt[1+i] for i in range(len(line_list))]
    err_velocities = [err_popt[1+i] for i in range(len(line_list))]
    sigs = [velocity_to_sig(line_list[i][1], popt[1+i])
            for i in range(len(line_list))]
    err_sigs = [velocity_to_sig(line_list[i][1], popt[1+i] + err_popt[1+i]) - sigs[i]
                for i in range(len(line_list))]

    amps = popt[3:] / scale_factor
    err_amps = err_popt[3:] / scale_factor
    flux_tuples = [get_flux(amps[i], sigs[i], amp_err=err_amps[i], sig_err=err_sigs[
                            i]) for i in range(len(line_list))]
    fluxes = [i[0] for i in flux_tuples]
    err_fluxes = [i[1] for i in flux_tuples]
    ha_idx = [idx for idx, name in enumerate(
        line_names) if name == 'Halpha'][0]
    pab_idx = [idx for idx, name in enumerate(line_names) if name == 'PaBeta'][0]

    # Apply nii correction to the halpha flux
    import copy
    nii_cor_fluxes = copy.deepcopy(fluxes)
    nii_cor_err_fluxes = copy.deepcopy(err_fluxes)
    nii_correction_factor = get_nii_correction(save_name, sps_df = [])
    nii_cor_fluxes[ha_idx] = nii_cor_fluxes[ha_idx] * nii_correction_factor
    nii_cor_err_fluxes[ha_idx] = nii_cor_err_fluxes[ha_idx] * nii_correction_factor


    ha_pab_ratio = [nii_cor_fluxes[ha_idx] / nii_cor_fluxes[pab_idx] for i in range(len(line_list))]
    eq_widths = [nii_cor_fluxes[i] / cont_values[i] for i in range(len(line_list))]
    ha_pab_av = [compute_ha_pab_av(1/ha_pab_ratio[i]) for i in range(len(line_list))]

    
    def compute_percentile_errs_on_line(line_idx, measured_line_flux, nii_cor=False):
        line_amps = [arr_popt[i][1 + len(line_list) + line_idx]/scale_factor for i in range(len(arr_popt))]
        line_sigs = [velocity_to_sig(line_list[line_idx][1], arr_popt[i][1+line_idx])for i in range(len(arr_popt))]
        line_fluxes = [get_flux(line_amps[i], line_sigs[i])[0] for i in range(len(arr_popt))]
        if nii_cor == True:
            line_fluxes = [line_fluxes[i] * nii_correction_factor for i in range(len(line_fluxes))]
        err_line_fluxes_low_high = np.percentile(line_fluxes, [16, 84])
        err_line_fluxes_low_high = np.abs(measured_line_flux-err_line_fluxes_low_high)
        
        return line_fluxes, err_line_fluxes_low_high


    def list_compute_err_and_logerr(fluxes_num, fluxes_denom):
        flux_list = [compute_err_and_logerr(fluxes_num[i], fluxes_denom[i], -99, -99) for i in range(len(fluxes_num))]
        value_outs = [flux_list[i][0] for i in range(len(flux_list))]
        log_value_outs = [flux_list[i][1] for i in range(len(flux_list))]
        return value_outs, log_value_outs
    
    err_fluxes = -99*np.ones(len(ha_pab_ratio))
    err_fluxes_low = -99*np.ones(len(ha_pab_ratio))
    err_fluxes_high = -99*np.ones(len(ha_pab_ratio))
    nii_cor_err_fluxes = -99*np.ones(len(ha_pab_ratio))
    nii_cor_err_fluxes_low = -99*np.ones(len(ha_pab_ratio))
    nii_cor_err_fluxes_high = -99*np.ones(len(ha_pab_ratio))
    err_amps = -99*np.ones(len(ha_pab_ratio))
    err_sigs = -99*np.ones(len(ha_pab_ratio))
    err_velocity_low = -99*np.ones(len(ha_pab_ratio))
    err_velocity_high = -99*np.ones(len(ha_pab_ratio))
    err_velocity = -99*np.ones(len(ha_pab_ratio))


    if n_loops > 0:
        all_ha_fluxes, hg_errs_low_high = compute_percentile_errs_on_line(ha_idx, fluxes[ha_idx])
        all_nii_cor_ha_fluxes, nii_cor_hg_errs_low_high = compute_percentile_errs_on_line(ha_idx, nii_cor_fluxes[ha_idx], nii_cor=True)
        all_pab_fluxes, hd_errs_low_high = compute_percentile_errs_on_line(pab_idx, fluxes[pab_idx])
        all_ha_pab_ratios = [all_nii_cor_ha_fluxes[i]/all_pab_fluxes[i] for i in range(len(arr_popt))]
        all_avs = [compute_ha_pab_av(1/all_ha_pab_ratios[i]) for i in range(len(arr_popt))]

        velocity_monte_carlo = [arr_popt[i][1] for i in range(len(arr_popt))]
        err_velocity_low_high = np.percentile(velocity_monte_carlo, [16,84])
        
        err_velocity_low = -99*np.ones(len(ha_pab_ratio))
        err_velocity_high = -99*np.ones(len(ha_pab_ratio))
        err_velocity = np.mean([err_velocity_low, err_velocity_high], axis=0)

        # Ha to be in line_index order
        err_fluxes = [np.mean(hg_errs_low_high), np.mean(hd_errs_low_high)]
        err_fluxes_low = [hg_errs_low_high[0], hd_errs_low_high[0]]
        err_fluxes_high = [hg_errs_low_high[1], hd_errs_low_high[1]]

        nii_cor_err_fluxes = [np.mean(nii_cor_hg_errs_low_high), np.mean(hd_errs_low_high)]
        nii_cor_err_fluxes_low = [nii_cor_hg_errs_low_high[0], hd_errs_low_high[0]]
        nii_cor_err_fluxes_high = [nii_cor_hg_errs_low_high[1], hd_errs_low_high[1]]


         
        monte_carlo_df = pd.DataFrame(zip(velocity_monte_carlo, all_ha_fluxes, all_nii_cor_ha_fluxes, all_pab_fluxes, all_ha_pab_ratios), columns = ['velocity', 'ha_flux', 'nii_cor_ha_flux', 'pab_flux', 'ha_pab_ratio'])
        imd.check_and_make_dir(emission_fit_dir)
        monte_carlo_df.to_csv(emission_fit_dir + f'{save_name}_monte_carlo.csv', index=False)


   
  
    err_ha_pab_ratio_low = ha_pab_ratio - np.percentile(all_ha_pab_ratios, 16)
    err_ha_pab_ratio_high = np.percentile(all_ha_pab_ratios, 84) - ha_pab_ratio
    err_av_low = ha_pab_av - np.percentile(all_avs, 16)
    err_av_high = np.percentile(all_avs, 84) - ha_pab_av

    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, err_z_offset, velocities, err_velocity, 
                              err_velocity_low, err_velocity_high, amps, err_amps, 
                              sigs, err_sigs, fluxes, err_fluxes, err_fluxes_low, err_fluxes_high, nii_cor_fluxes, nii_cor_err_fluxes, nii_cor_err_fluxes_low, nii_cor_err_fluxes_high, ha_pab_ratio, err_ha_pab_ratio_low, err_ha_pab_ratio_high, ha_pab_av, err_av_low, err_av_high, eq_widths), 
                              columns=['line_name', 'line_center_rest', 'z_offset', 'err_z_offset', 
                                       'velocity', 
                                       'err_fixed_velocity', 'err_fixed_velocity_low', 'err_fixed_velocity_high', 
                                       'amplitude', 'err_amplitude', 'sigma', 'err_sigma', 'flux', 'err_flux', 'err_flux_low', 'err_flux_high', 'nii_cor_flux', 'err_nii_cor_flux', 'err_nii_cor_flux_low', 'err_nii_cor_flux_high', 'ha_pab_ratio', 'err_ha_pab_ratio_low', 'err_ha_pab_ratio_high', 'ha_pab_av', 'err_ha_pab_av_low', 'err_ha_pab_av_high', 'equivalent_width_aa'])
    fit_df['signal_noise_ratio'] = fit_df['flux']/fit_df['err_flux']

    imd.check_and_make_dir(emission_fit_dir)
    fit_df.to_csv(emission_fit_dir + f'/{save_name}_emission_fits.csv', index=False)
    plot_emission_fit(emission_fit_dir, save_name, spectrum)
    return



def plot_emission_fit(emission_fit_dir, save_name, total_spec_df, ax_plot='', plot_type=''):
    """Plots the fit to each emission line

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    axis_group (int): Set to the number of the axis ratio group to fit that instead
    scaled (str): Set to true if plotting the scaled fits
    run_name (str): Set to name of prospector run to fit with those
    bootstrap_num (int): Which number in the bootstrap to plot, -1 to plot the original
    plot_type (str): 'ha_only' or 'pab_only'

    Returns:
    Saves a pdf of the fits for all of the lines
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fit_df = ascii.read(emission_fit_dir + f'/{save_name}_emission_fits.csv').to_pandas()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.09, 0.08, 0.88, 0.42])
    ax_Ha = fig.add_axes([0.55, 0.55, 0.40, 0.40])
    ax_Hb = fig.add_axes([0.09, 0.55, 0.40, 0.40])
    if plot_type == 'ha_only':
        ax_Ha = ax_plot
    if plot_type == 'pab_only':
        ax_Hb = ax_plot

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

  
    # spectrum = total_spec_df['rest_flux_erg_aa']
    wavelength = total_spec_df['rest_wave_aa']
    continuum_df = ascii.read(emission_fit_dir+f'{save_name}_cont_sub.csv').to_pandas()
    continuum = continuum_df['continuum_sub_ydata']
    cont_wavelength = continuum_df['wavelength']


    # Set up the parameters from the fitting
    pars = []
    pars.append(fit_df['z_offset'].iloc[0])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['velocity'])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['amplitude'])

    gauss_fit = multi_gaussian(wavelength, pars, fit=False)
    gauss_fit_df = pd.DataFrame(zip(wavelength, gauss_fit), columns=['rest_wavelength', 'gaussian_fit'])
    gauss_fit_df.to_csv(emission_fit_dir + f'{save_name}_gaussian_fit.csv', index=False)
    hb_range = wavelength > 12000

    # Plots the spectrum and fit on all axes
    for axis in axes_arr:
        # axis.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
        axis.plot(cont_wavelength, continuum, color='black', lw=1, label='Continuum-Sub')
        axis.plot(gauss_fit_df[hb_range]['rest_wavelength'], gauss_fit_df[hb_range]['gaussian_fit'], color='orange',
                  lw=1, label='Gaussian Fit')
        axis.plot(gauss_fit_df[~hb_range]['rest_wavelength'], gauss_fit_df[~hb_range]['gaussian_fit'], color='orange',
                  lw=1)
        
        if axis != ax:
            # Add text for each of the lines:
            for i in range(len(line_list)):
                line_name = line_list[i][0]
                line_wave = line_list[i][1] + fit_df['z_offset'].iloc[0]
                line_idxs = np.logical_and(
                    wavelength > line_wave - 10, wavelength < line_wave + 10)
                # axis.text(line_wave - 10, np.max(spectrum[line_idxs]) * 1.02, line_name, fontsize=10)
                axis.plot([line_wave, line_wave], [-100, 100],
                          ls='--', alpha=0.5, color='mediumseagreen')

    

 
    Ha_plot_range = ha_fit_range  # Angstrom
    Hb_plot_range = pab_fit_range
    # Hb_plot_range = (4995, 5015)

    def set_plot_ranges(ax, axis, plot_range, box_color):
        lim_min = 0.9 * np.min(continuum[np.logical_and(cont_wavelength > plot_range[0], wavelength < plot_range[1])])
        lim_max = 1.05 * np.max(continuum[np.logical_and(cont_wavelength > plot_range[0], wavelength < plot_range[1])])
        axis.set_ylim(lim_min, lim_max)
        axis.set_xlim(plot_range)
        rect = patches.Rectangle((plot_range[0], lim_min), (plot_range[1] - plot_range[0]), (lim_max - lim_min), linewidth=1.5, edgecolor=box_color, facecolor='None')
        ax.add_patch(rect)

    set_plot_ranges(ax, ax_Ha, Ha_plot_range, Ha_zoom_box_color)
    set_plot_ranges(ax, ax_Hb, Hb_plot_range, Hb_zoom_box_color)

    ax_Ha.text(0.05, 0.93, f"Ha: {round(10**17*fit_df.iloc[0]['flux'], 4)}", transform=ax_Ha.transAxes)
    ax_Hb.text(0.05, 0.93, f"PaB: {round(10**17*fit_df.iloc[1]['flux'], 4)}", transform=ax_Hb.transAxes)
    ax_Ha.text(0.05, 0.83, f"Ratio: {round(fit_df.iloc[0]['ha_pab_ratio'], 4)}", transform=ax_Ha.transAxes)
    ax_Hb.text(0.05, 0.83, f"Ratio: {round(fit_df.iloc[1]['ha_pab_ratio'], 4)}", transform=ax_Hb.transAxes)
    ax_Ha.text(0.95, 0.93, f"SNR: {round(fit_df.iloc[0]['signal_noise_ratio'], 4)}", transform=ax_Ha.transAxes, horizontalalignment='right')
    ax_Hb.text(0.95, 0.93, f"SNR: {round(fit_df.iloc[1]['signal_noise_ratio'], 4)}", transform=ax_Hb.transAxes, horizontalalignment='right')

    ax.set_ylim(-1 * 10**-20, 2e-19)
    # ax.set_ylim(np.percentile(continuum, [1, 99]))
    

    ax.legend(loc=1, fontsize=axisfont - 3)

    ax.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    if plot_type == '':
        fig.savefig(emission_fit_dir + 'plots' +
                    f'/{save_name}_emission_fit.pdf')
    plt.close(fig)
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


def monte_carlo_fit(func, wavelength, continuum, y_data, y_err, guess, bounds, n_loops, fit_axis_group=0, fast_continuum_cut=0):
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

    y_data_cont_sub = subtract_continuum(y_data, continuum)

    # y_data = y_data.fillna(0)
    # y_data_cont_sub = y_data_cont_sub.fillna(0)

    start = time.time()
    popt, pcov = curve_fit(func, wavelength, y_data_cont_sub, guess, bounds=bounds)
    end = time.time()
    print(f'Length of one fit: {end-start}')
    start = time.time()

    start = time.time()
    # Create a new set of y_datas

    #Fill over nan values with the median error * 3
    # y_err = y_err.fillna(5*y_err.median())

    new_y_datas = [[np.random.normal(loc=y_data.iloc[j], scale=y_err.iloc[j]) for j in range(len(y_data))] for i in range(n_loops)]
    
    # Turn them into dataframes with matching indicies
    new_y_data_dfs = [pd.DataFrame(new_y, columns=['flux']).set_index(y_data.index)['flux'] for new_y in new_y_datas]
    
    # Scale and subtract the continuum of of each
    new_y_datas_cont_sub = [subtract_continuum(new_y_data_dfs[i], continuum) for i in range(len(new_y_data_dfs))]

    fits_out = [curve_fit(func, wavelength, new_y, guess, bounds=bounds) for new_y in new_y_datas_cont_sub]

    new_popts = [i[0] for i in fits_out]

    end = time.time()
    print(f'Length of {n_loops} fits: {end-start}')

    return popt, new_popts, y_data_cont_sub


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
def multi_gaussian(wavelength, *pars, fit=True):
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

    velocities = [pars[1+i] for i in range(len(line_list))]
    amps = [pars[1+i+len(line_list)] for i in range(len(line_list))]
    
    line_names = [line_list[i][0] for i in range(len(line_list))]

    # Split the wavelength into its Halpha nad Pabeta parts
    wavelength_ha = wavelength[wavelength < 10000]
    wavelength_pab = wavelength[wavelength > 10000]


    #start_2 = time.time()
    gaussians_ha = gaussian_func(wavelength_ha, line_list[0][1] + z_offset, amps[0], velocity_to_sig(line_list[0][1], velocities[0]))
    gaussians_pab = gaussian_func(wavelength_pab, line_list[1][1] + z_offset, amps[1], velocity_to_sig(line_list[1][1], velocities[1]))

    
    combined_gauss = np.concatenate([gaussians_ha, gaussians_pab])
    y_vals = combined_gauss

    # # Do I need to separate the wavelngth sections? Seems to be useing the whole range for both?
    # gaussians = [gaussian_func(wavelength, line_list[i][1] + z_offset, amps[i], velocity_to_sig(line_list[i][1], velocities[i])) for i in range(len(line_list))]
    
    # y_vals = np.sum(gaussians, axis=0)

    return y_vals




def subtract_continuum(y_data, continuum):
    """Scales the continuum around the h_alpha and h_beta regions independently,k the n returns a subtracted version

    Parameters:
    y_data (array): flux/spectrum data, cut to only the h_alpha and h_beta regions
    continuum (array): continuum values, cut similarly

    Returns:
    y_data_cont_sub (array): Continuum subtracted y_data, only in the regions around h_alpha and h_beta
    """
    y_data = y_data - continuum

    return y_data


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


# def fit_continuum(wavelength, flux):
#     median_spectrum = np.median(flux)
#     clipped_spec = flux < 2*np.median(flux)
#     regress_res = linregress(wavelength[clipped_spec], flux[clipped_spec])
#     x_regress = wavelength
#     continuum = regress_res.intercept + regress_res.slope*x_regress
#     return continuum

def fit_continuum(wavelength, flux, plot_cont=True, save_name=''):
    # combined_mask = clip_elines(flux, wavelength)
    ha_region = wavelength < 10000
    pab_region = ~ha_region
    # ha_region_mask = wavelength[combined_mask] < 10000
    # pab_region_mask = wavelength[combined_mask] > 10000

    # ha_eline_mask = clip_elines_findpeaks(flux[ha_region], wavelength[ha_region])
    # pab_eline_mask = clip_elines_findpeaks(flux[pab_region], wavelength[pab_region])
    ha_eline_mask = mask_elines_known_lines(flux[ha_region], wavelength[ha_region])
    pab_eline_mask = mask_elines_known_lines(flux[pab_region], wavelength[pab_region])
    combined_mask = ha_eline_mask + pab_eline_mask
    # combined_mask = mask_lines(combined_mask, wavelength, line_list)    
    
    ha_regress_res = linregress(wavelength[combined_mask][ha_region], flux[combined_mask][ha_region])
    pab_regress_res = linregress(wavelength[combined_mask][pab_region], flux[combined_mask][pab_region])
    continuum = ha_regress_res.intercept + ha_regress_res.slope*wavelength
    continuum[pab_region] = pab_regress_res.intercept + pab_regress_res.slope*wavelength[pab_region]
    
   
    if plot_cont:
        fig, axarr = plt.subplots(1,2,figsize=(12,6))
        ax_ha = axarr[0]
        ax_pab = axarr[1]
        def plot_cont_axis(ax, region):
            ax.plot(wavelength[region], flux[region], color='red', label='masked', marker='o')
            ax.plot(wavelength[combined_mask][region], flux[combined_mask][region], color='black', label='use', marker='o', ls='None')
            ax.plot(wavelength[region], continuum[region], color='orange', label='continuum')
            ax.set_xlabel('Wavelength ($\\AA$)')
            ax.set_ylabel('Flux')
        plot_cont_axis(ax_ha, ha_region)
        plot_cont_axis(ax_pab, pab_region)
        ax_ha.legend()
        fig.savefig(f'/Users/brianlorenz/uncover/Data/emission_fitting/continuum/{save_name}_cont.pdf')
        plt.close()
    return continuum

def mask_elines_known_lines(flux, wavelength):
    mask_ha = np.logical_or(wavelength < 6200, wavelength > 7000)
    mask_pab = np.logical_or(wavelength < 12200, wavelength > 13400)
    mask = np.logical_and(mask_ha, mask_pab)
    mask = mask.tolist()
    return mask 

def clip_elines_findpeaks(flux, wavelength):
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(flux, prominence=(np.std(flux), None))
    mask = np.full(len(flux), True)
    mask[peaks] = False
    mask = mask.tolist()
    mask = extend_mask(mask)
    mask = extend_mask(mask)
    return mask

def clip_elines(flux, wavelength):
    check_range = 10
    mask = []
    for i in range(len(flux)):
        if i < check_range:
            low_range = 0
            high_range = 2*check_range
        elif i > len(flux)-check_range:
            low_range = -2*check_range
            high_range = -1
        else:
            low_range = i - check_range
            high_range = i + check_range
        flux_in_region = np.median(flux[low_range:high_range])
        if flux.iloc[i] > 1.5 * flux_in_region:
            mask.append(False)
        else:
            mask.append(True)
    mask =  extend_mask(mask)
    mask =  extend_mask(mask)
    mask = mask_lines(mask, wavelength, line_list)
    return mask

def extend_mask(mask1):
    mask2 = []
    for i in range(len(mask1)):
        if i == 0 or i == len(mask1)-1:
            mask2.append(True)
        elif mask1[i-1] == False:
            mask2.append(False)
        elif mask1[i+1] == False:
            mask2.append(False)
        elif mask1[i] == False:
            mask2.append(False)
        else:
            mask2.append(True)
    return mask2

def mask_lines(mask, wavelength, line_list):
    line_width = 200 #angstrom
    for i in range(len(line_list)):  
        line_wave = line_list[i][1]
        low_bound = line_wave - line_width
        high_bound = line_wave + line_width
        line_mask = np.logical_or(wavelength<low_bound, wavelength>high_bound)
        mask = np.logical_and(mask, line_mask)
    return mask


def get_fit_range(wavelength):
    """Gets the arrray of booleans that contains the two ranges to perform fitting

    Parameters:
    wavelength (pd.DataFrame): Dataframe of wavelength

    Returns:
    """
    cut_ha = np.logical_and(
        wavelength > ha_fit_range[0], wavelength < ha_fit_range[1])
    cut_pab = np.logical_and(
        wavelength > pab_fit_range[0], wavelength < pab_fit_range[1])
    full_cut = np.logical_or(cut_pab, cut_ha)
    return full_cut





def fit_all_emission_uncover(id_msa_list):
    for id_msa in id_msa_list:
        if id_msa < 39745:
            continue
        spec_df = read_fluxcal_spec(id_msa)
        print(f'Fitting emission for {id_msa}')
        fit_emission_uncover(spec_df, id_msa)


def plot_mosaic(id_msa_list, line = 'ha_only'):
    "line (str): 'ha_only' or 'pab_only' "
    nrows = 4
    ncols = 4
    fig, axarr = plt.subplots(nrows, ncols, figsize=(20,20))
    plot_idxs = []
    plot_count = 0
    for i in range(nrows):
        for j in range(ncols):
            plot_idxs.append([i, j])
    for id_msa in id_msa_list:
        ax = axarr[plot_idxs[plot_count][0], plot_idxs[plot_count][1]]
        spec_df = read_fluxcal_spec(id_msa)
        plot_emission_fit(emission_fit_dir, id_msa, spec_df, ax_plot=ax, plot_type=line)
        plot_count = plot_count + 1
        scale_aspect(ax)
        ax.set_title(f'id_msa = {id_msa}', fontsize=18)
    while plot_count < nrows*ncols:
        ax = axarr[plot_idxs[plot_count][0], plot_idxs[plot_count][1]]
        ax.set_xticks([])
        ax.set_yticks([])
        plot_count = plot_count + 1

    fig.savefig(emission_fit_dir + 'plots' + f'/mosaic_{line}.pdf')


if __name__ == "__main__":
    # # (Currently using)
    # id_msa = 18471
    # id_msa = 19179
    # id_msa = 14573
    id_msa = 50000
    spec_df = read_fluxcal_spec(id_msa)
    fit_emission_uncover(spec_df, id_msa)

    # # Fitting the mock spectra
    # mock_name = 'mock_ratio_15_flat'
    # spec_df = ascii.read(f'/Users/brianlorenz/uncover/Data/mock_spectra/{mock_name}.csv').to_pandas()
    # fit_emission_uncover(spec_df, mock_name)


    # id_msa_list = get_id_msa_list(full_sample=False)
    
    # fit_all_emission_uncover(id_msa_list)  
    # plot_mosaic(id_msa_list, line = 'ha_only')
    # plot_mosaic(id_msa_list, line = 'pab_only')
    pass