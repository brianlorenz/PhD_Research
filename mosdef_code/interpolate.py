# Contains functions that perform interpolation on an SED

import sys
import os
import string
import pdb
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
from read_data import mosdef_df
from mosdef_obj_data_funcs import get_mosdef_obj, read_sed, read_composite_sed
from plot_funcs import populate_main_axis
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd


np.random.seed(seed=901969)

# Generate the 20 evenly spaced (in log space) filters
log_filter_centers = np.linspace(np.log10(1300), np.log10(20000), 20)
width = 0.25


def gen_mock_sed(field, v4id, log_filter_centers=log_filter_centers, width=width, groupID=-99):
    """Create a mock SED at standard wavelengths

    Parameters:
    field (string): name of the field of the object
    v4id (int): HST V4.1 id of the object
    log_filter_centers (list): peak wavelengths of the centers, in log space
    width (float): size in log space to search for points around the filter center
    groupID (int): Set to the groupID of the composite to make a mock composite SED

    Returns:
    """

    if groupID > -1:
        sed = read_composite_sed(groupID)
        sed['err_f_lambda'] = (sed['err_f_lambda_u'] +
                               sed['err_f_lambda_d']) / 2

    else:
        sed = read_sed(field, v4id)
        mosdef_obj = get_mosdef_obj(field, v4id)
        sed['rest_wavelength'] = sed['peak_wavelength'] / \
            (1 + mosdef_obj['Z_MOSFIRE'])

    # Only consider those measurements with detections? Or do we allow less
    # than 0 fluxes? Set them to zero?
    good_idxs = np.logical_and(
        sed['f_lambda'] > -99, sed['err_f_lambda'] >= 0)

    # SED that only contains the good indexes, use this for fitting
    sed_fit = sed[good_idxs]

    # Assumes all fiters are boxes that run into each other and are the same
    # size
    filter_size = (log_filter_centers[1] - log_filter_centers[0]) / 2

    mock_sed = []
    mock_sed_u_errs = []
    mock_sed_l_errs = []
    mock_wave_centers = []

    for log_center_wave in log_filter_centers:
        # Pull all rows within +- width of peak wavelength
        lower_wave = 10**(log_center_wave - width)
        upper_wave = 10**(log_center_wave + width)
        points = sed_fit[sed_fit['rest_wavelength'].between(
            lower_wave, upper_wave)]
        unused_points = sed_fit[np.logical_not(sed_fit['rest_wavelength'].between(
            lower_wave, upper_wave))]

        while len(points[(points['rest_wavelength'] - 10**log_center_wave) > 0]) < 3:
            # Offsets from the central wavelength, dropping any negative valeus
            diffs = unused_points[(unused_points['rest_wavelength'] -
                                   10**log_center_wave) > 0]['rest_wavelength'] - 10**log_center_wave
            # Try to add the nearest point. If this hits a Value Error, it
            # should be because the sequence is empty, so there are no points
            # to add
            try:
                add_index = diffs.idxmin()
                # Update the points and unused_points objects accordingly
                points = points.append(sed_fit.loc[add_index])
                unused_points = unused_points.drop(add_index)
            except ValueError:
                break

        # Same as above, but other direction
        while len(points[(10**log_center_wave - points['rest_wavelength']) > 0]) < 3:
            # Offsets from the central wavelength, dropping any negative valeus
            diffs = 10**log_center_wave - unused_points[(10**log_center_wave - unused_points['rest_wavelength'])
                                                        > 0]['rest_wavelength']
            # Try to add the nearest point. If this hits a Value Error, it
            # should be because the sequence is empty, so there are no points
            # to add
            try:
                add_index = diffs.idxmin()
                # Update the points and unused_points objects accordingly
                points = points.append(sed_fit.loc[add_index])
                unused_points = unused_points.drop(add_index)
            except ValueError:
                break

        # If there aren't enough points, skip over this and append null values
        if len(points) < 2:
            mock_sed.append(-99)
            mock_sed_u_errs.append(-99)
            mock_sed_l_errs.append(-99)
            mock_wave_centers.append(10**log_center_wave)
            continue

        # If there is not data on both sides of the point, skip over this and
        # append null values
        if np.min(points['rest_wavelength']) > 10**log_center_wave or np.max(points['rest_wavelength']) < 10**log_center_wave:
            mock_sed.append(-99)
            mock_sed_u_errs.append(-99)
            mock_sed_l_errs.append(-99)
            mock_wave_centers.append(10**log_center_wave)
            continue

        # First, fit the points
        coeff = np.polyfit(np.log10(points['rest_wavelength']),
                           points['f_lambda'], deg=2, w=(1 / points['err_f_lambda']))
        # Get the polynomial
        fit_func = np.poly1d(coeff)
        # x-range over which we fit
        fit_wavelengths = np.arange(
            np.log10(lower_wave), np.log10(upper_wave), 0.001)
        # Values of the points we fit
        fit_points = fit_func(fit_wavelengths)
        # Indexes of the values that lie in the mock filter
        fit_idx = np.logical_and(fit_wavelengths > (log_center_wave -
                                                    filter_size), fit_wavelengths < (log_center_wave + filter_size))
        # Average the values in the mock filter to get the mock point
        mock_sed_point = np.mean(fit_points[fit_idx])
        # Append the final point to the list for the mock sed
        mock_sed_point, mock_sed_u_err, mock_sed_l_err = fit_uncertainty(
            points, lower_wave, upper_wave, log_center_wave, filter_size)
        mock_sed.append(mock_sed_point)
        mock_sed_u_errs.append(mock_sed_u_err)
        mock_sed_l_errs.append(mock_sed_l_err)
        mock_wave_centers.append(10**log_center_wave)
    vis_fit(field, v4id, sed, points, mock_wave_centers, width,
            fit_wavelengths, fit_points, filter_size, mock_sed, mock_sed_u_errs, mock_sed_l_errs, good_idxs, groupID)
    sed_df = pd.DataFrame(np.transpose([10**log_filter_centers, mock_sed,
                                        mock_sed_u_errs, mock_sed_l_errs]), columns=['rest_wavelength', 'f_lambda', 'err_f_lambda_u', 'err_f_lambda_d'])
    if groupID > -1:
        sed_df.to_csv(imd.mock_composite_sed_csvs_dir + f'/{groupID}_mock_sed.csv', index=False)
    else:
        sed_df.to_csv(imd.home_dir + f'/mosdef/mock_sed_csvs/{field}_{v4id}_sed.csv', index=False)
    return None


def fit_uncertainty(points, lower_wave, upper_wave, log_center_wave, filter_size):
    """Performs fitting many times to get an estimate of the uncertainty

    """
    mock_points = []
    for i in range(1, 100):
        # First, fit the points
        coeff = np.polyfit(np.log10(points['rest_wavelength']),
                           np.random.normal(points['f_lambda'], points['err_f_lambda']), deg=2)  # , w=(1/points['err_f_lambda'])
        # Get the polynomial
        fit_func = np.poly1d(coeff)
        # x-range over which we fit
        fit_wavelengths = np.arange(
            np.log10(lower_wave), np.log10(upper_wave), 0.001)
        # Values of the points we fit
        fit_points = fit_func(fit_wavelengths)
        # Indexes of the values that lie in the mock filter
        fit_idx = np.logical_and(fit_wavelengths > (log_center_wave -
                                                    filter_size), fit_wavelengths < (log_center_wave + filter_size))
        # Average the values in the mock filter to get the mock point
        mock_sed_point = np.mean(fit_points[fit_idx])
        mock_points.append(mock_sed_point)
    # PERCENTILE ERROR HERE?
    mock_sed_point, l_err, u_err = np.percentile(mock_points, [50, 15.7, 84.3])
    return mock_sed_point, u_err - mock_sed_point, mock_sed_point - l_err


def vis_fit(field, v4id, sed, points, mock_wave_centers, width, fit_wavelengths, fit_points, filter_size, mock_sed, mock_sed_u_errs, mock_sed_l_errs, good_idxs, groupID, showfit=False):
    """Visualizes the fitting to the SED

    Parameters:
    sed (pd.DataFrame): full SED that you are fitting
    points (pd.DataFrame): subset of points that are currently highlighted
    mock_wave_centers (float): center wavelengths of all computed so far
    width (float): size in log space to search for points around the filter center
    fit_wavelengths (list, np.poly1d): wavelengths that were fit
    fit_points(): values of the fit
    filter_size: how wide the mock filter is (half distance to next filter)
    mock_sed(list): list of all mock points that have been computed so far
    mock_sed_u_errs(list): upper errors corresponding to the points
    mock_sed_l_errs(list): lower errors corresponding to the points
    good_idxs(): Points where data is above -99

    Returns:
    """

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig, ax = plt.subplots(figsize=(8, 8))
    populate_main_axis(ax, sed, good_idxs, axisfont, ticksize, ticks)

    if showfit:
        ax.axvspan(10**(np.log10(mock_wave_centers[-1]) - filter_size),
                   10**(np.log10(mock_wave_centers[-1]) + filter_size), facecolor='grey', alpha=0.5, label='Filter')
        ax.plot(10**fit_wavelengths, fit_points,
                color='mediumseagreen', label='Fit')

    ax.errorbar(mock_wave_centers, mock_sed, yerr=[mock_sed_u_errs, mock_sed_l_errs],
                color='blue', label='Mock SED', ls='None', marker='o')

    if groupID > -1:
        ax.text(0.02, 0.95, f'Cluster: {groupID}', fontsize=axisfont, transform=ax.transAxes)
    else:
        ax.text(0.02, 0.95, f'{field} {v4id}', fontsize=axisfont, transform=ax.transAxes)

    ax.set_ylim(min(0, min(sed[good_idxs]['f_lambda'])) * 1.2,
                max(sed[good_idxs]['f_lambda'] * 1.2))
    ax.set_xscale('log')
    ax.set_xlim(800, 45000)
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=legendfont, loc=1)
    if groupID > -1:
        fig.savefig(imd.mock_composite_sed_images_dir + f'/{groupID}_mock.pdf')
    else:
        fig.savefig(imd.home_dir + f'/mosdef/SED_Images/mock_sed_images/{field}_{v4id}_mock.pdf')
    plt.close('all')
    return None


def gen_all_seds(zobjs):
    """Given a field and id, plots the SED of a galaxy from its {field}_{v4id}_sed.csv

    Parameters:
    zobjs (list): Pass a list of tuples of the form (field, v4id)


    Returns:
    """
    counter = 1
    for obj in zobjs:
        field = obj[0]
        v4id = obj[1]
        print(f'Creating SED for {field}_{v4id}, {counter}/{len(zobjs)}')
        if v4id == -9999:
            print('v4id = -9999')
            counter = counter + 1
            continue
        try:
            gen_mock_sed(field, v4id)
        except Exception as excpt:
            print(f'Couldnt create SED for {field}_{v4id}')
            print(excpt)
            plt.close('all')
            sys.exit()
        counter = counter + 1


def gen_all_mock_composites(n_clusters):
    """Given a field and id, plots the SED of a galaxy from its {field}_{v4id}_sed.csv

    Parameters:
    n_clusters (int): Number of clusters


    Returns:
    """
    for groupID in range(n_clusters):
        print(f'Creating mock SED for {groupID}')
        try:
            gen_mock_sed('0', 0, groupID=groupID)
        except Exception as excpt:
            print(f'Couldnt create SED for Cluster {groupID}')
            print(excpt)
            plt.close('all')
            sys.exit()
