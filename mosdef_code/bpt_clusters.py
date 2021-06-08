# Deals with the BPT of the seds and composites.

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
from emission_measurements import read_emission_df, get_emission_measurements
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf


def get_bpt_coords(emission_df, zobjs):
    """Gets the row(s) corresponding to one object

    Parameters:
    emission_df (pd.DataFrame): Dataframe containing emission line measurements and info
    zobjs (list): list of tuples of the form (field, v4id)

    Returns:
    bpt_df (pd.DataFrame): Dataframe of values to plot on the bpt diagram
    """

    # Will be used to store results
    fields = []
    v4ids = []
    log_NII_Has = []
    log_NII_Ha_errs = []
    log_OIII_Hbs = []
    log_OIII_Hb_errs = []

    for zobj in zobjs:
        # Read in the emissionn lines for this object
        elines = get_emission_measurements(emission_df, zobj)
        # Re-name them
        Ha_flux, Ha_err = elines['HA6565_FLUX'], elines['HA6565_FLUX_ERR']
        NII_flux, NII_err = elines['NII6585_FLUX'], elines['NII6585_FLUX_ERR']
        Hb_flux, Hb_err = elines['HB4863_FLUX'], elines['HB4863_FLUX_ERR']
        OIII_flux, OIII_err = elines[
            'OIII5008_FLUX'], elines['OIII5008_FLUX_ERR']
        NII_Ha, NII_Ha_err = elines['NIIHA'], elines['NIIHA_ERR']

        # Check for zeros:
        fluxes_errs = np.array(
            [Ha_flux.iloc[0], Ha_err.iloc[0], NII_flux.iloc[0], NII_err.iloc[0], Hb_flux.iloc[0], Hb_err.iloc[0], OIII_flux.iloc[0], OIII_err.iloc[0]])
        # If any less than zero, set them all to -99
        if (fluxes_errs <= 0).any():
            fields.append(zobj[0])
            v4ids.append(zobj[1])
            log_NII_Has.append(-99)
            log_NII_Ha_errs.append(-99)
            log_OIII_Hbs.append(-99)
            log_OIII_Hb_errs.append(-99)
            continue

        # Calculate ratios and uncertainties
        # log_NII_Ha, log_NII_Ha_err = calc_log_ratio(
        #     NII_flux, NII_err, Ha_flux, Ha_err)
        log_NII_Ha = np.log10(NII_Ha)
        log_NII_Ha_err = 0.434 * (NII_Ha_err / NII_Ha)
        NII_Ha_err = elines['NIIHA'], elines['NIIHA_ERR']
        log_OIII_Hb, log_OIII_Hb_err = calc_log_ratio(
            OIII_flux, OIII_err, Hb_flux, Hb_err)

        # Append the results
        fields.append(zobj[0])
        v4ids.append(zobj[1])
        log_NII_Has.append(log_NII_Ha.iloc[0])
        log_NII_Ha_errs.append(log_NII_Ha_err.iloc[0])
        log_OIII_Hbs.append(log_OIII_Hb.iloc[0])
        log_OIII_Hb_errs.append(log_OIII_Hb_err.iloc[0])

    # Compile all resutls into a dataframe
    bpt_df = pd.DataFrame(zip(fields, v4ids, log_NII_Has, log_NII_Ha_errs, log_OIII_Hbs, log_OIII_Hb_errs), columns=['field', 'v4id',
                                                                                                                     'log_NII_Ha', 'log_NII_Ha_err', 'log_OIII_Hb', 'log_OIII_Hb_err'])
    # Return the dataframe
    return bpt_df


def calc_log_ratio(top_flux, top_err, bot_flux, bot_err):
    """Calculates np.log10(top/bot) and its uncertainty

    Parameters:
    Fluxes an errors for each of the lines

    Returns:
    log_ratio (float): np.log10(top/bot)
    log_ratio_err (float): uncertainty in np.log10(top/bot)
    """
    log_ratio = np.log10(top_flux / bot_flux)
    log_ratio_err = (1 / np.log(10)) * (bot_flux / top_flux) * np.sqrt(
        ((1 / bot_flux) * top_err)**2 + ((-top_flux / (bot_flux**2)) * bot_err)**2)
    return log_ratio, log_ratio_err


def plot_bpt(emission_df, zobjs, savename='None', axis_obj='False', composite_bpt_point=[-47], composite_bpt_errs=0):
    """Plots the bpt diagram for the objects in zobjs

    Parameters:
    emission_df (pd.DataFrame): Dataframe containing emission line measurements and info
    zobjs (list): list of tuples of the form (field, v4id)
    savename (str): location with name ot save the file
    axis_obj (matplotlib_axis): Replace with an axis to plot on an existing axis
    composite_bpt_point (): Set to the point if using a composite sed and you want to plot the bpt point of that

    Returns:
    """

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    # Get the bpt valeus to plot for all object
    bpt_df = get_bpt_coords(emission_df, zobjs)

    if axis_obj == 'False':
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        ax = axis_obj

    # Bpt diagram lines
    xline = np.arange(-3.0, 0.469, 0.001)
    yline = 0.61 / (xline - 0.47) + 1.19  # Kewley (2001)
    xlineemp = np.arange(-3.0, 0.049, 0.001)
    ylineemp = 0.61 / (xlineemp - 0.05) + 1.3  # Kauffman (2003)
    ax.plot(xline, yline, color='dimgrey', lw=2,
            ls='--', label='Kewley+ (2001)')
    ax.plot(xlineemp, ylineemp, color='dimgrey',
            lw=2, ls='-', label='Kauffmann+ (2003)')

    for i in range(len(bpt_df)):
        gal = bpt_df.iloc[i]
        ax.errorbar(gal['log_NII_Ha'], gal['log_OIII_Hb'], xerr=gal[
                    'log_NII_Ha_err'], yerr=gal['log_OIII_Hb_err'], marker='o', color='black', ecolor='grey')

    if composite_bpt_point[0] != -47:
        ax.errorbar(composite_bpt_point[0], composite_bpt_point[
            1], xerr=np.array([composite_bpt_errs[0]]).T, yerr=np.array([composite_bpt_errs[1]]).T, marker='o', color='red', ecolor='red')

    ax.set_xlim(-2, 1)
    ax.set_ylim(-1.2, 1.5)

    if axis_obj == 'False':
        ax.set_xlabel('log(N[II] 6583 / H$\\alpha$)', fontsize=axisfont)
        ax.set_ylabel('log(O[III] 5007 / H$\\beta$)', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)

    if savename != 'None':
        fig.savefig(savename)

    plt.close('allplo')


def plot_bpt_clusters(n_clusters):
    """Plots the bpt diagram for every cluster

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    # Read in the emission lines dataframe
    emission_df = read_emission_df()

    for groupID in range(n_clusters):
        # Get the names of all galaxies in the cluster
        cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)
        fields_ids = [(obj[0], int(obj[1])) for obj in fields_ids]
        # Location to save the file
        savename = imd.mosdef_dir + f'/Clustering/cluster_stats/bpt_diagrams/{groupID}_BPT.pdf'
        plot_bpt(emission_df, fields_ids, savename=savename)
