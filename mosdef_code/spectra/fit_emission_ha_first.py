# Fit halpha emission, get a veloicty dispersion, then fit hbeta using that velocity

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
from fit_emission import compute_bootstrap_uncertainties

halpha_wave = 6564.61
hbeta_wave = 4862.68
line_list = [
    ('Halpha', 6564.61),
    ('Hbeta', 4862.68)
]
line_centers_rest = [line_list[i][1] for i in range(len(line_list))]


def fit_emission_ha_first(groupID, norm_method, bootstrap_num=-1):
    """Given a groupID, fit the emission lines in that composite spectrum

    Parameters:
    groupID (int): Number of the cluster to fit
    norm_methd (str): Method used for normalization, points to the folder where spectra are stored
    bootstrap_num (int): Set to -1 to avoid bootstrap, set to the number to read in the corresponding spectrum and fit that

    Returns:
    Saves a csv of the fits for all of the lines
    """
    # Number of loops in Monte Carlo
    n_loops = 0

    
    composite_spectrum_df = read_composite_spectrum(groupID, norm_method, bootstrap_num=bootstrap_num)

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
    # Only applying guess for 1 line (halpha), that why it's -1
    for i in range(len(line_list)-1):
        guess.append(scale_factor * amp_guess)
        bounds_low.append(0)
        bounds_high.append(scale_factor * 10**-16)
    bounds = (np.array(bounds_low), np.array(bounds_high))

    wavelength = composite_spectrum_df[
        'wavelength']
    continuum = composite_spectrum_df[
            'cont_f_lambda']
    full_cut = get_fit_range(wavelength)
    wavelength_cut = wavelength[full_cut]
    continuum_cut = continuum[full_cut]
    
    
    z_offset_ha, velocity_ha, amplitude_ha, amplitude_hb, y_data_cont_sub, hb_scale, ha_scale  = monte_carlo_fit(wavelength_cut, scale_factor * continuum_cut, scale_factor * composite_spectrum_df[full_cut]['f_lambda'], scale_factor * composite_spectrum_df[full_cut]['err_f_lambda'], np.array(guess), bounds, n_loops)
    
    # Save continuum-subtracted ydata
    imd.check_and_make_dir(imd.emission_fit_dir + '/ha_first_cont_subs/')
    cont_sub_df = pd.DataFrame(zip(wavelength_cut, y_data_cont_sub / scale_factor), columns=['wavelength_cut','continuum_sub_ydata'])
    cont_sub_df.to_csv(imd.emission_fit_dir + '/ha_first_cont_subs' + f'/{groupID}_cont_sub.csv', index=False)

    # Now, parse the results into a dataframe
    hb_scales = [hb_scale for i in range(len(line_list))]
    ha_scales = [ha_scale for i in range(len(line_list))]
    line_names = [line_list[i][0] for i in range(len(line_list))]
    line_centers_rest = [line_list[i][1] for i in range(len(line_list))]
    z_offset = [z_offset_ha for i in range(len(line_list))]
    err_z_offset = [-99 for i in range(len(line_list))]
    velocity = [velocity_ha for i in range(len(line_list))]
    err_velocity = [-99 for i in range(len(line_list))]
    sigs = [velocity_to_sig(line_list[i][1], velocity_ha)
            for i in range(len(line_list))]
    err_sigs = [-99 for i in range(len(line_list))]

    amps = [amplitude_ha / scale_factor, amplitude_hb / scale_factor]
    err_amps = [-99 for i in range(len(line_list))]
    flux_tuples = [get_flux(amps[i], sigs[i], amp_err=err_amps[i], sig_err=err_sigs[
                            i]) for i in range(len(line_list))]
    fluxes = [i[0] for i in flux_tuples]
    ha_flux = fluxes[0]
    hb_flux = fluxes[1]
    err_fluxes = [i[1] for i in flux_tuples]
    
    balmer_dec = [ha_flux / hb_flux for i in range(len(line_list))]
    err_balmer_dec_low = [-99 for i in range(len(line_list))]
    err_balmer_dec_high = [-99 for i in range(len(line_list))]



    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, err_z_offset, hb_scales, ha_scales, velocity, err_velocity, amps, err_amps, sigs, err_sigs, fluxes, err_fluxes, balmer_dec, err_balmer_dec_low, err_balmer_dec_high), columns=['line_name', 'line_center_rest', 'z_offset', 'err_z_offset', 'hb_scale', 'ha_scale', 'fixed_velocity', 'err_fixed_velocity', 'amplitude', 'err_amplitude', 'sigma', 'err_sigma', 'flux', 'err_flux', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high'])
    fit_df['signal_noise_ratio'] = fit_df['flux']/fit_df['err_flux']


    
    if bootstrap_num > -1:
        imd.check_and_make_dir(imd.emission_fit_dir + '/ha_first_boot_csvs/')
        fit_df.to_csv(imd.emission_fit_dir + f'/ha_first_boot_csvs/{groupID}_emission_fits_{bootstrap_num}.csv', index=False)
    else:
        imd.check_and_make_dir(imd.emission_fit_dir + '/ha_first_csvs/')
        imd.check_and_make_dir(imd.emission_fit_dir + '/ha_first_images/')
        fit_df.to_csv(imd.emission_fit_dir + '/ha_first_csvs/' +
                    f'/{groupID}_emission_fits.csv', index=False)
        plot_emission_fit_ha_first(groupID, norm_method)
    return




def plot_emission_fit_ha_first(groupID, norm_method, axis_group=-1, save_name='', scaled='False', run_name='False', bootstrap_num=-1):
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


    fit_df = ascii.read(imd.emission_fit_dir + '/ha_first_csvs/' +
                    f'/{groupID}_emission_fits.csv').to_pandas()
    total_spec_df = read_composite_spectrum(groupID, norm_method)
    cont_sub_df = ascii.read(imd.emission_fit_dir + '/ha_first_cont_subs/' + f'{groupID}_cont_sub.csv').to_pandas()

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
    continuum = total_spec_df['cont_f_lambda']
    wavelength = total_spec_df['wavelength']
    

    too_low_gals, plot_cut, not_plot_cut, n_gals_in_group, cutoff, cutoff_low, cutoff_high = get_too_low_gals(
        groupID, norm_method, save_name, axis_group=axis_group)

    # Set up the parameters from the fitting
    pars = []
    pars.append(fit_df['z_offset'].iloc[0])
    pars.append(fit_df['fixed_velocity'].iloc[0])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['amplitude'])

    full_cut = get_fit_range(wavelength)
    wavelength_cut = wavelength[full_cut]
    hb_half_idx = wavelength_cut < 5500
    ha_half_idx = np.logical_not(hb_half_idx)
    hb_cut = get_cuts(wavelength_cut[hb_half_idx])
    ha_cut = get_cuts(wavelength_cut[ha_half_idx])

    
    gauss_fit_ha = single_gaussian_ha(wavelength_cut[ha_half_idx], pars[0], pars[1], pars[2])
    
    def single_gaussian_hb(wavelength, amplitude):
        gauss = gaussian_func(wavelength, hbeta_wave + pars[0], amplitude, velocity_to_sig(hbeta_wave, pars[1]))
        return gauss 
    gauss_fit_hb = single_gaussian_hb(wavelength_cut[hb_half_idx], pars[3])
    
    gauss_fit_df_hb = pd.DataFrame(zip(wavelength_cut[hb_half_idx], gauss_fit_hb), columns=['rest_wavelength', 'gaussian_fit'])
    gauss_fit_df_ha = pd.DataFrame(zip(wavelength_cut[ha_half_idx], gauss_fit_ha), columns=['rest_wavelength', 'gaussian_fit'])
    gauss_fit_df = pd.concat([gauss_fit_df_hb, gauss_fit_df_ha], ignore_index=True,axis=0)
    imd.check_and_make_dir(imd.emission_fit_dir + f'/ha_first_gaussian_fits/')
    gauss_fit_df.to_csv(imd.emission_fit_dir + f'/ha_first_gaussian_fits/{groupID}_gaussian_fit.csv', index=False)
   

    # Plots the continuum-subtracted spectrum and fit on all axes
    for axis in axes_arr:
        # axis.plot(wavelength, spectrum, color='black', lw=1, label='Cont-sub Spectrum')
        axis.plot(cont_sub_df['wavelength_cut'], cont_sub_df['continuum_sub_ydata'], color='black', label='Continuum-Subtracted', marker='None', ls='-')
       
        axis.plot(gauss_fit_df['rest_wavelength'], gauss_fit_df['gaussian_fit'], color='orange',
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
        lim_min = 0.9 * np.min(cont_sub_df['continuum_sub_ydata'])
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

    fig.savefig(imd.emission_fit_dir +
                f'/ha_first_images/{groupID}_emission_fit.pdf')
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


def monte_carlo_fit(wavelength_cut, continuum, y_data, y_err, guess, bounds, n_loops, fit_axis_group=0, fast_continuum_cut=0):
    '''Fit the multi-gaussian to the data, use monte carlo to get uncertainties

    Parameters:
   
    Returns:
    popt (list): List of the fit parameters
    err_popt (list): Uncertainties on these parameters
    '''
    # Ranges for ha and hb
    hb_half_idx = wavelength_cut < 5500
    ha_half_idx = np.logical_not(hb_half_idx)
    
    hb_cut = get_cuts(wavelength_cut[hb_half_idx])
    ha_cut = get_cuts(wavelength_cut[ha_half_idx])

    y_data_cont_sub, hb_scale, ha_scale = scale_continuum(y_data, continuum, hb_half_idx, ha_half_idx, hb_cut, ha_cut)

    y_data = y_data.fillna(0)
    y_data_cont_sub = y_data_cont_sub.fillna(0)

    start = time.time()

    popt_ha, pcov_ha = curve_fit(single_gaussian_ha, wavelength_cut[ha_half_idx], y_data_cont_sub[ha_half_idx], guess, bounds=bounds)
    z_offset_ha = popt_ha[0]
    velocity_ha = popt_ha[1]
    amplitude_ha = popt_ha[2]
    
    def single_gaussian_hb(wavelength, amplitude):
        gauss = gaussian_func(wavelength, hbeta_wave + z_offset_ha, amplitude, velocity_to_sig(hbeta_wave, velocity_ha))
        return gauss 

    popt_hb, pcov_hb = curve_fit(single_gaussian_hb, wavelength_cut[hb_half_idx], y_data_cont_sub[hb_half_idx], 1, bounds=(0, 100))
    end = time.time()
    amplitude_hb = popt_hb[0]
    print(f'Length of one fit: {end-start}')


    return z_offset_ha, velocity_ha, amplitude_ha, amplitude_hb, y_data_cont_sub, hb_scale, ha_scale 


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


def single_gaussian_ha(wavelength, z_offset, velocity, amplitude):

    gauss = gaussian_func(wavelength, halpha_wave + z_offset, amplitude, velocity_to_sig(halpha_wave, velocity))
    return gauss 

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

def fit_all_group_emission_ha_first(n_clusters, ignore_groups, n_boots):
    for groupID in range(n_clusters):
        if groupID in ignore_groups:
            continue
        fit_emission_ha_first(groupID, 'cluster_norm', bootstrap_num=-1)
    # Then, fit all the boostrapped emission
    for groupID in range(n_clusters):
        if groupID in ignore_groups:
            continue
        for i in range(n_boots):
            fit_emission_ha_first(groupID, 'cluster_norm', bootstrap_num=i)


# fit_all_group_emission_ha_first(23, ignore_groups=[19], n_boots=100)
compute_bootstrap_uncertainties(23, '', bootstrap=100, clustering=True, ignore_groups=[19], ha_first=True)