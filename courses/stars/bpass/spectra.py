

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from hoki import load
from scipy import interpolate

import scipy.integrate as integrate

bpass_dir = '/Users/galaxies-air/bpass/BPASSv2.2.1_release-07-18-Tuatara/bpass_v2.2.1_imf135_100/'


def read_bpass(binary=True, z='020', bpass_dir=bpass_dir):
    """Uses hoki to read a spectra file from BPASS

    Parameters:
    binary (boolean): whether to use a 'bin' or 'sin' file
    z (str): metallicity. '020' is solar
    bpass_dir (str): directory where the pass spectra are located

    Returns:
    spec_df (pd.DataFrame): hoki pandas dataframe of the spectrum corresponing to z and binary
    """
    if binary:
        usebin = 'bin'
    else:
        usebin = 'sin'
    spectra_file = f'spectra-{usebin}-imf135_100.z{z}.dat'

    spec_df = load.model_output(bpass_dir+spectra_file)
    return spec_df


def get_normalized_sed(spec1_df, spec2_df, log_age_1, log_age_2):
    """Normalize an SED (changes) to a target SED (unchanged)

    Parameters:
    spec1_df (pd.DataFrame): First spectrum, this is the standard to normalize to
    spec2_df (pd.DataFrame): Second spectrum, this is the one that gets normalized
    log_age_1 (str): age to compare the spectra at, in the form 'X.X' (or 'XX.X' for ages >= 10.0)
    log_age_2 (str): age to compare the spectra at, in the form 'X.X' (or 'XX.X' for ages >= 10.0)

    Returns:
    spec_norm (pd.DataFrame): a single_column dataframe with the normalized fluxes of spec2_df[log_age]
    """

    # Computes the normalization factor a12, need to multiply this by flux of spec2 to normalize it
    norm_factor = np.sum(
        spec1_df[log_age_1]*spec2_df[log_age_2]) / np.sum(spec2_df[log_age_2]**2)

    print(f'Normalizing by multiplying {norm_factor}')
    spec_norm = spec2_df[log_age_2]*norm_factor
    return spec_norm


def calc_flux_diff(spec1_df, spec2_df, log_age_1, log_age_2):
    """Given two spectra, normalize them and compute the difference in flux

    Parameters:
    spec1_df (pd.DataFrame): First spectrum, this is the standard to normalize to
    spec2_df (pd.DataFrame): Second spectrum, this is the one that gets normalized
    log_age_1 (str): age to compare the spectra at, in the form 'X.X' (or 'XX.X' for ages >= 10.0)
    log_age_2 (str): age to compare the spectra at, in the form 'X.X' (or 'XX.X' for ages >= 10.0)

    Returns:
    flux_diff (pd.DataFrame): values of the fluxes for the subtracted spectra
    spec_norm (pd.DataFrame): a single_column dataframe with the normalized fluxes of spec2
    """

    # Normalize the spectra:
    spec_norm = get_normalized_sed(spec1_df, spec2_df, log_age_1, log_age_2)
    #flux_diff = spec1_df[log_age_1]-spec_norm
    # New method of flux diff, trying to model out
    flux_diff = (spec1_df[log_age_1]-spec_norm)/(spec1_df[log_age_1]+spec_norm)
    #flux_diff = (spec1_df[log_age_1]-spec_norm)
    # breakpoint()
    return flux_diff, spec_norm


def plot_flux_diff(binary_spec_df, single_spec_df, log_age_1, log_age_2):
    """Makes the plot for first draft, difference in fluxes for two normalized spectra as a function of lambda

    Parameters:
    binary_spec_df (pd.DataFrame): dataframe containing the binary spectrum
    single_spec_df (pd.DataFrame): dataframe containing the single spectrum
    log_age_1 (str): age to compare the spectra at, in the form 'X.X' (or 'XX.X' for ages >= 10.0)
    log_age_2 (str): age to compare the spectra at, in the form 'X.X' (or 'XX.X' for ages >= 10.0)

    Returns:
    """

    # Compute the flux difference and noramlized fluxes
    flux_diff, spec_norm = calc_flux_diff(
        binary_spec_df, single_spec_df, log_age_1, log_age_2)

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Generate the first figure, which will just show the spectra
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot the two spectra, with the second normalized
    ax.plot(binary_spec_df['WL'], binary_spec_df[log_age_1],
            ls='-', marker='', color='black', label=f'Binary, Z$_\odot$, log(Age)={log_age_1}')
    ax.plot(single_spec_df['WL'], spec_norm,
            ls='-', marker='', color='blue', label=f'Single, Z$_\odot$, log(Age)={log_age_2}, Normalized', alpha=0.5)
    ax.plot(single_spec_df['WL'], single_spec_df[log_age_2],
            ls='-', marker='', color='red', label=f'Single, Z$_\odot$, log(Age)={log_age_2}, Unnormalized', alpha=0.2)

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('Flux', fontsize=axisfont)
    ax.set_xscale('log')
    ax.legend(fontsize=legendfont-4)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(f'/Users/galaxies-air/bpass/figures/spec_norm_{log_age_1}_{log_age_2}.png')
    fig.savefig(f'/Users/galaxies-air/bpass/figures/spec_norm_{log_age_1}_{log_age_2}.pdf')
    plt.close()

    '''

    '''
    # Second figure, with flux difference
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot the two spectra, with the second normalized
    ax.plot(binary_spec_df['WL'], flux_diff,
            ls='-', marker='', color='black')

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('Normalized Flux Difference', fontsize=axisfont)
    ax.set_xscale('log')
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(f'/Users/galaxies-air/bpass/figures/flux_diff_{log_age_1}_{log_age_2}.png')
    fig.savefig(f'/Users/galaxies-air/bpass/figures/flux_diff_{log_age_1}_{log_age_2}.pdf')
    plt.close()


def plot_flux_diff_time(binary_spec_df, single_spec_df, ages=['6.0', '7.0', '8.0', '9.0', '10.0']):
    """Makes the plot for first draft, difference in fluxes for two normalized spectra as a function of lambda and age

    Parameters:
    binary_spec_df (pd.DataFrame): dataframe containing the binary spectrum
    single_spec_df (pd.DataFrame): dataframe containing the single spectrum
    ages (list): list of ages to compare the spectra at

    Returns:
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig, ax = plt.subplots(figsize=(8, 7))

    # Compute the flux difference and noramlized fluxes
    flux_diffs = []
    spec_norms = []
    #color = iter(cm.infr(np.linspace(0, 0.8, len(ages))))
    color = iter(['black', 'blue', 'mediumseagreen', 'orange', 'red'])
    for age in ages:
        c = next(color)
        flux_diff, spec_norm = calc_flux_diff(
            binary_spec_df, single_spec_df, age, age)
        ax.plot(binary_spec_df['WL'], flux_diff,
                ls='-', marker='', color=c, label=f'log(Age)={age}')

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('d$_{12}$ (Binary - Single)', fontsize=axisfont)
    ax.set_xscale('log')
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=axisfont-2)
    fig.savefig(f'/Users/galaxies-air/bpass/figures/flux_diff_time.png')
    fig.savefig(f'/Users/galaxies-air/bpass/figures/flux_diff_time.pdf')
    plt.close()


# binary_spec_df = read_bpass(binary=True, z='020')
# single_spec_df = read_bpass(binary=False, z='020')

# # Filter the data to remove the UV (where flux is effectively 0)
# binary_spec_df = binary_spec_df[(binary_spec_df.WL > 500)]
# single_spec_df = single_spec_df[(single_spec_df.WL > 500)]

# # plot_flux_diff(binary_spec_df, single_spec_df, log_age_1='9.0', log_age_2='9.0')
# plot_flux_diff_time(binary_spec_df, single_spec_df)
