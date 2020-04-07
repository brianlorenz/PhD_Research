# Deals with the UVJ of the seds and composites

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
from clustering import cluster_dir
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate


def get_uvj(field, v4id):
    """Get the U-V and V-J for a given galaxy

    Parameters:
    field (str): field of the galaxy
    v4id (int): v4id from 3DHST

    Returns:
    uvj_tuple (tuple): tuple of the form (U-V, V-J) for the input object from mosdef
    """

    # Read the file
    uvj_df = ascii.read(
        '/Users/galaxies-air/mosdef/uvj_latest.dat').to_pandas()

    # Get the object from mosdef_df, since we need id and not v4id
    mosdef_obj = get_mosdef_obj(field, v4id)

    # Get the input object
    obj = uvj_df[np.logical_and(
        uvj_df['field'] == field, uvj_df['id'] == mosdef_obj['ID'])]

    # Get the U-V and V-J for that object
    u_v = obj['u_v'].iloc[0]
    v_j = obj['v_j'].iloc[0]
    uvj_tuple = (u_v, v_j)
    return uvj_tuple


def observe_composite_uvj(groupID):
    """Measure the U-V and V-J flux of a composite SED

    Parameters:
    groupID (int): groupid of the cluster that you want to measure the U-V and V-J for

    Returns:
    uvj_tuple (tuple): tuple of the form (U-V, V-J) for the input composite SED
    """

    sed = read_composite_sed(groupID)

    # Create an interpolation object. Use this with interp_sed(wavelength) to get f_lambda at that wavelength
    interp_sed = interpolate.interp1d(sed['rest_wavelength'], sed['f_lambda'])

    # Filters are U=153, V=155, J=161
    U_filt_num = 153
    V_filt_num = 155
    J_filt_num = 161

    # Fluxes are in f_nu NOT f_lambda
    U_flux_nu = observe_filt(interp_sed, U_filt_num)
    V_flux_nu = observe_filt(interp_sed, V_filt_num)
    J_flux_nu = observe_filt(interp_sed, J_filt_num)
    print(U_flux_nu, V_flux_nu, J_flux_nu)

    U_V = -2.5*np.log10(U_flux_nu/V_flux_nu)
    V_J = -2.5*np.log10(V_flux_nu/J_flux_nu)

    uvj_tuple = (U_V, V_J)
    return uvj_tuple


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
    denominator = integrate.quad(lambda wave: (wave /
                                               interp_filt(wave)), wavelength_min, wavelength_max)[0]
    flux_filter_nu = numerator/denominator
    return flux_filter_nu
