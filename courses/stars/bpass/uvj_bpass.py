# Deals with the UVJ of the seds and composites

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from filter_response import lines, overview, get_index, get_filter_response
from clustering import cluster_dir
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from spectra import read_bpass


def observe_uvj(sed, age_str):
    """Measure the U-V and V-J flux of a composite SED

    Parameters:
    sed (pd.DataFrame): sed to observe, needs columns 'rest_wavelength' and 'f_lambda'
    age_str (str): e.g. '7.0'

    Returns:
    uvj_tuple (tuple): tuple of the form (U-V, V-J) for the input composite SED
    """

    # Create an interpolation object. Use this with interp_sed(wavelength) to get f_lambda at that wavelength
    interp_sed = interpolate.interp1d(sed['WL'], sed[age_str])

    # Filters are U=153, V=155, J=161
    U_filt_num = 153
    V_filt_num = 155
    I_filt_num = 159
    J_filt_num = 161

    # Fluxes are in f_nu NOT f_lambda
    U_flux_nu = observe_filt(interp_sed, U_filt_num)
    V_flux_nu = observe_filt(interp_sed, V_filt_num)
    I_flux_nu = observe_filt(interp_sed, I_filt_num)
    J_flux_nu = observe_filt(interp_sed, J_filt_num)
    print(U_flux_nu, V_flux_nu, J_flux_nu)

    U_V = -2.5*np.log10(U_flux_nu/V_flux_nu)
    V_I = -2.5*np.log10(V_flux_nu/I_flux_nu)
    V_J = -2.5*np.log10(V_flux_nu/J_flux_nu)

    uvj_tuple = (U_V, V_J)
    return uvj_tuple, V_I


def observe_filt(interp_sed, filter_num):
    """given an SED filter interpolated, measure the value of the SED in that filter

    Parameters:
    interp_sed (scipy.interp1d): interp1d of the SED that you want to measure
    filter_num (int): Number of the filter to observe from in FILTER.RES.latest

    Returns:
    flux_filter_nu (int): The photometric SED point for that filter - the observation (in frequency units)
    """
    filter_df = get_filter_response(filter_num)[1]

    interp_filt = interpolate.interp1d(
        filter_df['wavelength'], filter_df['transmission'])

    wavelength_min = np.min(filter_df['wavelength'])
    wavelength_max = np.max(filter_df['wavelength'])
    numerator = integrate.quad(lambda wave: (1/3**18)*(wave*interp_sed(wave) *
                                                       interp_filt(wave)), wavelength_min, wavelength_max)[0]
    denominator = integrate.quad(lambda wave: (
        interp_filt(wave) / wave), wavelength_min, wavelength_max)[0]
    flux_filter_nu = numerator/denominator
    return flux_filter_nu


def plot_uvj_cluster(sed, age_strs):
    """given a groupID, plot the UVJ diagram of the composite and all galaxies within cluster

    Parameters:
    age_strs (list of strings): ['6.0','7.0','8.0'...]

    Returns:
    """

    # Get their uvj values
    uvjs = []
    for age_str in age_strs:
        uvjs.append(observe_uvj(sed, age_str))
        # Catalog calculation
        #uvjs.append(get_uvj(field, v4id))
    u_v = [i[0] for i in uvjs]
    v_j = [i[1] for i in uvjs]

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Generate the first figure, which will just show the spectra
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot the two spectra, with the second normalized
    ax.plot(v_j, u_v,
            ls='', marker='o', markersize=3, color='black', label='Cluster Galaxies')

    ax.set_xlabel('V-J', fontsize=axisfont)
    ax.set_ylabel('U-V', fontsize=axisfont)
    #ax.set_xlim(0, 2)
    #ax.set_ylim(0, 2.5)
    ax.legend(fontsize=legendfont-4)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig('/Users/galaxies-air/bpass/figures/UVJ.pdf')
    plt.close()


def plot_color_age(sed_bin, sed_sin, age_strs, uvjs_bin=0, uvjs_sin=0):
    """given a groupID, plot the UVJ diagram of the composite and all galaxies within cluster

    Parameters:
    age_strs (list of strings): ['6.0','7.0','8.0'...]

    Returns:
    """

    # Get their uvj values

    if uvjs_bin == 0:
        uvjs_bin = []
        v_is_bin = []
        for age_str in age_strs:
            uvj, v_i = observe_uvj(sed_bin, age_str)
            uvjs_bin.append(uvj)
            v_is_bin.append(v_i)
            # Catalog calculation
            #uvjs.append(get_uvj(field, v4id))
    u_v_bin = [i[0] for i in uvjs_bin]
    v_j_bin = [i[1] for i in uvjs_bin]

    if uvjs_bin == 0:
        uvjs_sin = []
        v_is_sin = []
        for age_str in age_strs:
            uvj, v_i = observe_uvj(sed_sin, age_str)
            uvjs_sin.append(uvj)
            v_is_sin.append(v_i)
            # Catalog calculation
            #uvjs.append(get_uvj(field, v4id))
    u_v_sin = [i[0] for i in uvjs_sin]
    v_j_sin = [i[1] for i in uvjs_sin]

    ages = [float(age_str) for age_str in age_strs]

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Generate the first figure, which will just show the spectra
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot the two spectra, with the second normalized
    # ax.plot(ages, v_is_bin,
    #         ls='', marker='o', markersize=4, color='black', label='Binary, Z=Z$_\odot$')
    # ax.plot(ages, v_is_sin,
    #         ls='', marker='o', markersize=4, color='red', label='Single, Z=Z$_\odot$')

    ax.plot([v_j_bin, v_j_sin], [u_v_bin, u_v_sin], ls='-',
            marker='', color='black')
    cax = ax.scatter(v_j_bin, u_v_bin, c=ages, label='Binary, Z=Z$_\odot$')
    ax.scatter(v_j_sin, u_v_sin, marker='x',
               c=ages, label='Single, Z=Z$_\odot$')
    cb = plt.colorbar(cax)
    cb.set_label('log(Age)', fontsize=axisfont)
    cb.ax.tick_params(labelsize=ticksize, size=ticks)

    ax.plot((-100, 0.69), (1.3, 1.3), color='red')
    ax.plot((1.5, 1.5), (2.01, 100), color='red')
    xline = np.arange(0.69, 1.5, 0.001)
    yline = xline*0.88+0.69
    ax.plot(xline, yline, color='red', label='Empirical Division')

    # ax.set_xlabel('log(Age)', fontsize=axisfont)
    # ax.set_ylabel('V-I', fontsize=axisfont)
    ax.set_xlabel('V-J', fontsize=axisfont)
    ax.set_ylabel('U-V', fontsize=axisfont)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=legendfont, loc=2)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig('/Users/galaxies-air/bpass/figures/uvj_age.pdf')
    plt.close()


# binary_spec_df = read_bpass(binary=True, z='020')
# single_spec_df = read_bpass(binary=False, z='020')

# lowz
# binary_spec_df = read_bpass(binary=True, z='002')
# single_spec_df = read_bpass(binary=False, z='002')

# highz
#binary_spec_df = read_bpass(binary=True, z='040')
#single_spec_df = read_bpass(binary=False, z='040')

# # Filter the data to remove the UV (where flux is effectively 0)
# binary_spec_df = binary_spec_df[(binary_spec_df.WL > 500)]
# single_spec_df = single_spec_df[(single_spec_df.WL > 500)]

# age_strs = ['6.0', '7.0', '8.0', '9.0', '10.0']

# age_strs = ['6.0',  '6.1',  '6.2',  '6.3',  '6.4',  '6.5',  '6.6',  '6.7',  '6.8',  '6.9',  '7.0',
#             '7.1',  '7.2',  '7.3',  '7.4',  '7.5',  '7.6', '7.7',  '7.8',  '7.9',  '8.0',  '8.1',
#             '8.2',  '8.3',  '8.4',  '8.5',  '8.6',  '8.7',  '8.8',  '8.9',  '9.0',  '9.1',  '9.2',
#             '9.3',  '9.4',  '9.5',  '9.6',  '9.7',  '9.8',  '9.9', '10.0']

# plot_color_age(binary_spec_df, single_spec_df, age_strs)

#plot_color_age(binary_spec_df, single_spec_df, age_strs, uvjs_bin, uvjs_sin)
