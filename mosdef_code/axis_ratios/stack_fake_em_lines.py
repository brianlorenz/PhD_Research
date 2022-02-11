# Generate fake emission lines with different velocity dispersions, do a median stack, then change the velocity dispersion, then stack again, talk to Aliza about this, N-med method? 
import numpy as np
import pandas as pd
from fit_emission import gaussian_func, velocity_to_sig, get_amp, get_flux
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import initialize_mosdef_dirs as imd
from scipy import interpolate
from stack_spectra import perform_stack
from scipy.optimize import curve_fit


def main(n_spec=1000):
    """
    Runs all the functions to generate the plot. 

    Parameters:
    n_spec (int): Number of spectra to generate
    """
    gal_dfs = generate_group_of_spectra(n_spec=n_spec)
    interp_spectrum_dfs = [interpolate_spec(gal_dfs[i]) for i in range(len(gal_dfs))]
    norm_factors = np.ones(len(interp_spectrum_dfs))
    median_total_spec, _, _, _, _ = perform_stack('median', interp_spectrum_dfs, norm_factors)
    mean_total_spec, _, _, _, _ = perform_stack('mean', interp_spectrum_dfs, norm_factors)
    plot_group_of_spectra(gal_dfs, interp_spectrum_dfs, median_total_spec, mean_total_spec)

def plot_group_of_spectra(gal_dfs, interp_spectrum_dfs, median_total_spec, mean_total_spec):
    """Plots the spectra passed to it
    
    """

    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    fig = plt.figure(constrained_layout=True, figsize=(10, 14))

    gs = GridSpec(5, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    spec_axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    ax_stack = fig.add_subplot(gs[3:, 0:])
    spec_wavelength = interp_spectrum_dfs[0]['rest_wavelength']

    for i in range(9):
        ax = spec_axs[i]
        ax.plot(gal_dfs[i]['rest_wavelength'], gal_dfs[i]['f_lambda_norm'], marker='o', color='black', ls='-', label = 'rest-frame')
        ax.plot(spec_wavelength, interp_spectrum_dfs[i]['f_lambda_norm'], marker='o', color='blue', ls='-', label='interpolated')
        ax.set_ylim(0, 0.5)
        ax.set_xlim(6553, 6577)
        ax.set_xlabel('wavelength')
        ax.set_ylabel('flux')

    ax_stack.plot(spec_wavelength, median_total_spec, color='black', marker='o', ls='-', label='median')
    ax_stack.plot(spec_wavelength, mean_total_spec, color='orange', marker='o', ls='-', label='mean')
    ax_stack.set_xlabel('wavelength')
    ax_stack.set_ylabel('flux')

    # Fit the output emission lines

    mean_center, mean_amp, mean_sigma, mean_line_flux = fit_emission_line(spec_wavelength, mean_total_spec, 6565)
    median_center, median_amp, median_sigma, median_line_flux = fit_emission_line(spec_wavelength, median_total_spec, 6565)


    ax1.legend(loc=1)
    ax_stack.legend(loc=1, fontsize=14)
    ax_stack.text(6573, 2.4, f'Stacked {len(gal_dfs)} Galaxies')
    fig.savefig(imd.axis_output_dir + f'/emline_stack_tests/stacked_{len(gal_dfs)}.pdf')
    plt.close('all')

def interpolate_spec(spectrum_df):
    """Interpolate teh spectrum and add uncertainties in the same way that is done in stack_spectra"""
    spectrum_wavelength = np.arange(6553, 6577, 0.5)
    norm_interp = interpolate.interp1d(
                spectrum_df['rest_wavelength'], spectrum_df['f_lambda_norm'], fill_value=0, bounds_error=False)
    spectrum_flux_norm = norm_interp(spectrum_wavelength)
    spectrum_err_norm = np.ones(len(spectrum_wavelength)) / 10
    cont_norm = np.zeros(len(spectrum_wavelength))
    interp_spectrum_df = pd.DataFrame(zip(spectrum_wavelength, spectrum_flux_norm, spectrum_err_norm, cont_norm), columns=['rest_wavelength', 'f_lambda_norm', 'err_f_lambda_norm', 'cont_norm'])
    return interp_spectrum_df

def generate_group_of_spectra(n_spec=9):
    """Makes a group of spectra from the parameter space

    Parameters:
    n_spec (int): Number of spectra to generate
    """
    
    zs = [np.random.random() + 1.6 for i in range(n_spec)]
    vel_disps = [np.random.random()*80 + 70 for i in range(n_spec)]
    gal_dfs = [generate_fake_galaxy_prop(zs[i], 1, vel_disps[i], 6565) for i in range(n_spec)]
    return gal_dfs

def generate_fake_galaxy_prop(z, flux, vel_disp, line_peak):
    """Makes a fake galaxy spectrum with the properties specified
    
    Parameters:
    z (float): redshift of the galaxy
    flux (float): flux of the line
    vel_disp (float): velocity dispersion of the galaxy 
    line_peak (float): Peak wavelength of the line to generate for
    """

    obs_line_peak = line_peak * (1+z)
    wavelength = np.arange(int(obs_line_peak)-40, int(obs_line_peak)+40, 1.5)
    fluxes = generate_emission_line(wavelength, flux, vel_disp)
    rest_wavelength = wavelength / (1+z)
    rest_flux = fluxes * (1+z)
    gal_df = pd.DataFrame(zip(wavelength, fluxes, rest_wavelength, rest_flux), columns = ['wavelength', 'flux', 'rest_wavelength', 'f_lambda_norm'])
    # fit_emission_line(gal_df, 6565)
    return gal_df
    


def generate_emission_line(wavelength, flux, vel_disp):
    """Given an amplitude and velocity dispersion, generate an emission line
    
    Parameters:
    wavelength (array): Wavelength array to put the line on. It will be centered
    flux (float): flux of the line
    vel_disp (float): Velocity dispersion of the line, will be converted into a width

    Returns:
    fluxes (array): Array of same length as wavelength, containing the line fluxes
    """
    # Put the line in the center of the wavelength range
    line_center = np.median(wavelength)

    # Convert the velocity dispersion into a line width, depends on wavelength
    sigma = velocity_to_sig(line_center, vel_disp)
    amplitude = get_amp(flux, sigma)
    # print(amplitude)
    fluxes = gaussian_func(wavelength, line_center, amplitude, sigma)
    return fluxes


def fit_emission_line(rest_wavelength, fluxes, line_center):
    """Fit an emission line from a gal_df

    Parameters:
    rest_wavelength (array): Wavelength array
    fluxes (array): Fluxes array
    line_center (float): Central wavelength (angstrom)
    
    """
    guess_center = line_center
    guess_amp = 0.15
    guess_sigma = 6
    guess = [guess_center, guess_amp, guess_sigma]
    popt, pcov = curve_fit(gaussian_func, rest_wavelength, fluxes, guess)
    center, amp, sigma = popt
    line_flux, _ = get_flux(amp, sigma)
    print(line_flux)
    return center, amp, sigma, line_flux

main()