import numpy as np
from stack_fake_em_lines import fit_emission_line
import matplotlib.pyplot as plt
from fit_emission import gaussian_func
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import pandas as pd
import os
import sys
from mosdef_obj_data_funcs import read_sed, get_mosdef_obj


def read_norm_specs(save_name, axis_group):
    spec_dir = imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spectra/{axis_group}'
    spec_files = os.listdir(spec_dir)
    spec_dfs = [ascii.read(spec_dir + '/' + filename).to_pandas() for filename in spec_files if '.csv' in filename]
    fields = [filename.split('_')[0] for filename in spec_files if '.csv' in filename]
    v4ids = [int(filename.split('_')[1]) for filename in spec_files if '.csv' in filename]
    mosdef_objs = [get_mosdef_obj(fields[i], v4ids[i]) for i in range(len(fields))]
    redshifts = [obj['Z_MOSFIRE_USE'] for obj in mosdef_objs]
    line_fluxes = []
    for i in range(len(spec_dfs)):
        line_fluxes.append(plot_norm_spec(spec_dfs[i], save_name, axis_group, spec_files[i]))

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(redshifts, line_fluxes, ls='None', marker='o', color='black')
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('Normalized Flux', fontsize=14)
    ax.tick_params(labelsize=14)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spec_plots/{axis_group}/cumulative_fig.pdf')
    plt.close('all')

    

def plot_norm_spec(spec_df, save_name, axis_group, filename):
    """Plots the normalized halpha line of the spectrum, fits it, then saves it by group, so we can compare the areas under the curves

    spectrum_wavelength (array): Wavelengths in angstroms
    norm_flux (array): fluxes in erg/s/cm^2/angstrom
    """
    spectrum_wavelength = spec_df['rest_wavelength']
    norm_flux = spec_df['f_lambda_norm']
    halpha_region = np.logical_and(spectrum_wavelength>6550, spectrum_wavelength<6580)
    halpha_waves = spectrum_wavelength[halpha_region]
    halpha_fluxes = norm_flux[halpha_region]*1e17
    center, amp, sigma, line_flux = fit_emission_line(halpha_waves, halpha_fluxes, 6563)
    fit_fluxes = gaussian_func(halpha_waves, center, amp, sigma)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(halpha_waves, halpha_fluxes, ls='None', marker='o', color='black', label='Interpolated cont')
    ax.plot(halpha_waves, fit_fluxes, ls='-', marker='None', color='orange', label='Fit flux')
    ax.set_title(f'Flux: {round(line_flux, 5)}', fontsize=14)
    ax.set_xlabel('Wavelength (Angstroms)', fontsize=14)
    ax.set_ylabel('Normalized Flux', fontsize=14)
    ax.tick_params(labelsize=14)
    imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spec_plots')
    imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spec_plots/{axis_group}')
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spec_plots/{axis_group}/{filename}.pdf')
    plt.close('all')
    return line_flux

read_norm_specs('both_ssfrs_4bin_mean', 0)