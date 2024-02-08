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

# Only usees this part of the spectrum for emission lines, continuum, fitting etc
fit_region = [4000, 4500]
emission_fit_dir = '/Users/brianlorenz/jwst_sfgalaxy/emission_fitting/'

line_list = [
    ('Hgamma', 4341.69),
    ('Hdelta', 4102.89)
]
lines_dict = {
    line_list[0][0]: line_list[0][1],
    line_list[1][0]: line_list[1][1]
}
line_centers_rest = [line_list[i][1] for i in range(len(line_list))]


def fit_emission(spectrum, save_name, bootstrap_num=-1):
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

    scale_factor = 10**18.5

    # Build the initial guesses
    guess = []
    bounds_low = []
    bounds_high = []
    amp_guess = 1  # flux units
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
        guess.append(amp_guess)
        bounds_low.append(0)
        bounds_high.append(100)
    bounds = (np.array(bounds_low), np.array(bounds_high))

    
    fit_idxs = np.logical_and(spectrum['rest_wavelength'] > fit_region[0], spectrum['rest_wavelength'] < fit_region[1]) 
    wavelength = spectrum[fit_idxs]['rest_wavelength']
    flux = spectrum[fit_idxs]['rest_flux_total_scaled']
    err_flux = spectrum[fit_idxs]['rest_err_total_scaled']
    continuum = fit_continuum(wavelength, flux)
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
    hg_idx = [idx for idx, name in enumerate(
        line_names) if name == 'Hgamma'][0]
    hd_idx = [idx for idx, name in enumerate(line_names) if name == 'Hdelta'][0]
    hg_hd_ratio = [fluxes[hg_idx] / fluxes[hd_idx] for i in range(len(line_list))]

    
    def compute_percentile_errs_on_line(line_idx, measured_line_flux):
        line_amps = [arr_popt[i][2 + line_idx]/scale_factor for i in range(len(arr_popt))]
        line_sigs = [velocity_to_sig(line_list[line_idx][1], arr_popt[i][
                               1])for i in range(len(arr_popt))]
        line_fluxes = [get_flux(line_amps[i], line_sigs[i])[0]
                     for i in range(len(arr_popt))]
        err_line_fluxes_low_high = np.percentile(line_fluxes, [16, 84])
        err_line_fluxes_low_high = np.abs(measured_line_flux-err_line_fluxes_low_high)
        
        return line_fluxes, err_line_fluxes_low_high


    def list_compute_err_and_logerr(fluxes_num, fluxes_denom):
        flux_list = [compute_err_and_logerr(fluxes_num[i], fluxes_denom[i], -99, -99) for i in range(len(fluxes_num))]
        value_outs = [flux_list[i][0] for i in range(len(flux_list))]
        log_value_outs = [flux_list[i][1] for i in range(len(flux_list))]
        return value_outs, log_value_outs
    
    err_fluxes = -99*np.ones(len(hg_hd_ratio))
    err_fluxes_low = -99*np.ones(len(hg_hd_ratio))
    err_fluxes_high = -99*np.ones(len(hg_hd_ratio))
    err_amps = -99*np.ones(len(hg_hd_ratio))
    err_sigs = -99*np.ones(len(hg_hd_ratio))
    err_velocity_low = -99*np.ones(len(hg_hd_ratio))
    err_velocity_high = -99*np.ones(len(hg_hd_ratio))
    err_velocity = -99*np.ones(len(hg_hd_ratio))

    if n_loops > 0:
        all_hg_fluxes, hg_errs_low_high = compute_percentile_errs_on_line(hg_idx, fluxes[hg_idx])
        all_hd_fluxes, hd_errs_low_high = compute_percentile_errs_on_line(hd_idx, fluxes[hd_idx])
        all_hg_hd_ratios = [all_hg_fluxes[i]/all_hd_fluxes[i] for i in range(len(arr_popt))]

        velocity_monte_carlo = [arr_popt[i][1] for i in range(len(arr_popt))]
        err_velocity_low_high = np.percentile(velocity_monte_carlo, [16,84])
        
        err_velocity_low = velocity - err_velocity_low_high[0]
        err_velocity_high = err_velocity_low_high[1] - velocity
        err_velocity = np.mean([err_velocity_low, err_velocity_high], axis=0)

        # Ha to be in line_index order
        err_fluxes = [np.mean(hg_errs_low_high), np.mean(hd_errs_low_high)]
        err_fluxes_low = [hg_errs_low_high[0], hd_errs_low_high[0]]
        err_fluxes_high = [hg_errs_low_high[1], hd_errs_low_high[1]]

         
        monte_carlo_df = pd.DataFrame(zip(velocity_monte_carlo, all_hg_fluxes, all_hd_fluxes, all_hg_hd_ratios), columns = ['velocity', 'hg_flux', 'hd_flux', 'hg_hd_ratio'])
        imd.check_and_make_dir(emission_fit_dir)
        monte_carlo_df.to_csv(emission_fit_dir + f'{save_name}_monte_carlo.csv', index=False)


   
  
    err_hg_hd_ratio_low = hg_hd_ratio - np.percentile(all_hg_hd_ratios, 16)
    err_hg_hd_ratio_high = np.percentile(all_hg_hd_ratios, 84) - hg_hd_ratio

    fit_df = pd.DataFrame(zip(line_names, line_centers_rest,
                              z_offset, err_z_offset, velocity, err_velocity, 
                              err_velocity_low, err_velocity_high, amps, err_amps, 
                              sigs, err_sigs, fluxes, err_fluxes, err_fluxes_low, err_fluxes_high, hg_hd_ratio, err_hg_hd_ratio_low, err_hg_hd_ratio_high), 
                              columns=['line_name', 'line_center_rest', 'z_offset', 'err_z_offset', 
                                       'fixed_velocity', 
                                       'err_fixed_velocity', 'err_fixed_velocity_low', 'err_fixed_velocity_high', 
                                       'amplitude', 'err_amplitude', 'sigma', 'err_sigma', 'flux', 'err_flux', 'err_flux_low', 'err_flux_high', 'hg_hd_ratio', 'err_hg_hd_ratio_low', 'err_hg_hd_ratio_high'])
    fit_df['signal_noise_ratio'] = fit_df['flux']/fit_df['err_flux']


    imd.check_and_make_dir(emission_fit_dir)
    fit_df.to_csv(emission_fit_dir + f'/{save_name}_emission_fits.csv', index=False)
    plot_emission_fit(emission_fit_dir, save_name, spectrum, fit_idxs)
    return



def plot_emission_fit(emission_fit_dir, save_name, total_spec_df, fit_idxs):
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

    fit_df = ascii.read(emission_fit_dir + f'/{save_name}_emission_fits.csv').to_pandas()

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

  
    spectrum = total_spec_df['rest_flux_total_scaled']
    wavelength = total_spec_df['rest_wavelength']
    continuum_df = ascii.read(emission_fit_dir+f'{save_name}_cont_sub.csv').to_pandas()
    continuum = continuum_df['continuum_sub_ydata']
    cont_wavelength = continuum_df['wavelength']



    # Set up the parameters from the fitting
    pars = []
    pars.append(fit_df['z_offset'].iloc[0])
    pars.append(fit_df['fixed_velocity'].iloc[0])
    for i in range(len(fit_df)):
        pars.append(fit_df.iloc[i]['amplitude'])

    gauss_fit = multi_gaussian(wavelength[fit_idxs], pars, fit=False)
    gauss_fit_df = pd.DataFrame(zip(wavelength[fit_idxs], gauss_fit), columns=['rest_wavelength', 'gaussian_fit'])
    gauss_fit_df.to_csv(emission_fit_dir + f'{save_name}_gaussian_fit.csv', index=False)
    hb_range = wavelength[fit_idxs] < 4150

    # Plots the spectrum and fit on all axes
    for axis in axes_arr:
        # axis.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
        axis.plot(cont_wavelength, continuum, color='black', lw=1, label='Continuum-Sub')
        axis.plot(wavelength[fit_idxs][hb_range], gauss_fit[hb_range], color='orange',
                  lw=1, label='Gaussian Fit')
        axis.plot(wavelength[fit_idxs][~hb_range], gauss_fit[~hb_range], color='orange',
                  lw=1)
        
        if axis != ax:
            # Add text for each of the lines:
            for i in range(len(line_list)):
                line_name = line_list[i][0]
                line_wave = line_list[i][1] + fit_df['z_offset'].iloc[0]
                line_idxs = np.logical_and(
                    wavelength > line_wave - 10, wavelength < line_wave + 10)
                axis.text(line_wave - 10, np.max(spectrum[line_idxs]) * 1.02, line_name, fontsize=10)
                axis.plot([line_wave, line_wave], [-100, 100],
                          ls='--', alpha=0.5, color='mediumseagreen')

    

 
    Ha_plot_range = (4320, 4380)  # Angstrom
    Hb_plot_range = (4070, 4130)
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

    ax_Ha.text(0.05, 0.93, f"Hg: {round(10**17*fit_df.iloc[0]['flux'], 4)}", transform=ax_Ha.transAxes)
    ax_Hb.text(0.05, 0.93, f"Hd: {round(10**17*fit_df.iloc[1]['flux'], 4)}", transform=ax_Hb.transAxes)
    ax_Hb.text(0.05, 0.83, f"Ratio: {round(fit_df.iloc[1]['hg_hd_ratio'], 4)}", transform=ax_Hb.transAxes)

    # ax.set_ylim(-1 * 10**-20, 1.01 * np.max(spectrum))
    ax.set_ylim(np.percentile(continuum, [1, 99]))

    ax.legend(loc=1, fontsize=axisfont - 3)

    ax.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    fig.savefig('/Users/brianlorenz/jwst_sfgalaxy/plots' +
                f'/{save_name}_emission_fit.pdf')
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
    velocity = pars[1]

    
    line_names = [line_list[i][0] for i in range(len(line_list))]


    gaussians = [gaussian_func(wavelength, line_list[i][1] + z_offset, pars[i + 2], velocity_to_sig(line_list[i][1], velocity)) for i in range(len(line_list))]
    
    y_vals = np.sum(gaussians, axis=0)

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


def fit_continuum(wavelength, flux):
    median_spectrum = np.median(flux)
    clipped_spec = flux < 2*np.median(flux)
    regress_res = linregress(wavelength[clipped_spec], flux[clipped_spec])
    x_regress = wavelength
    continuum = regress_res.intercept + regress_res.slope*x_regress
    return continuum


