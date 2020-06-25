import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
from read_data import mosdef_df
from mosdef_obj_data_funcs import get_mosdef_obj, read_sed
from polynomial_fit import poly_fit
import matplotlib.pyplot as plt
import fnmatch
import initialize_mosdef_dirs as imd
from spectra_funcs import clip_skylines, get_spectra_files


def populate_main_axis(ax, sed, good_idxs, axisfont, ticksize, ticks):
    ax.errorbar(sed[good_idxs]['rest_wavelength'], sed[good_idxs]['f_lambda'], yerr=[
                sed[good_idxs]['err_f_lambda'], sed[good_idxs]['err_f_lambda']], ls='None', color='black', marker='o')
    outliers = sed[np.logical_not(good_idxs)]
    [ax.axvspan(outliers.iloc[i]['rest_wavelength'] * 0.99,
                outliers.iloc[i]['rest_wavelength'] * 1.01, facecolor='r', alpha=0.5) for i in range(len(outliers))]
    ax.set_xlabel('log($\lambda$) ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('f$_{\lambda}$', fontsize=axisfont)


def plot_sed(field, v4id, plot_spec=False, plot_fit=False):
    """Given a field and id, read in the sed and create a plot of it

    Parameters:
    field (string): name of the field of the object
    v4id (int): HST V4.1 id of the object
    plot_spec (boolean): Set to true to plot spectrum
    plot_fit (boolean): Set to true to plot polynomial fit


    Returns:
    """
    sed = read_sed(field, v4id)
    mosdef_obj = get_mosdef_obj(field, v4id)
    sed['rest_wavelength'] = sed['peak_wavelength'] / \
        (1 + mosdef_obj['Z_MOSFIRE'])

    mosdef_obj = get_mosdef_obj(field, v4id)

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig = plt.figure(figsize=(8, 8))
    # Set the location of the image axis here
    ax_main = fig.add_axes([0.15, 0.5, 0.75, 0.45])
    ax_zoom = fig.add_axes([0.15, 0.1, 0.75, 0.25])
    ax_image = fig.add_axes([0.74, 0.79, 0.15, 0.15])

    # Values that are not outliers
    good_idxs = sed['f_lambda'] > -98
    populate_main_axis(ax_main, sed, good_idxs, axisfont, ticksize, ticks)
    populate_main_axis(ax_zoom, sed, good_idxs, axisfont, ticksize, ticks)
    # Bottom limit is either 0 or 1.2* most negative value. Upper limit is
    # 1.2* most positive value

    ax_main.set_ylim(min(0, min(sed[good_idxs]['f_lambda'])) * 1.2,
                     max(sed[good_idxs]['f_lambda'] * 1.2))
    ax_zoom.set_ylim(min(0, min(sed[good_idxs][sed[good_idxs]['peak_wavelength'] < 20000]['f_lambda'])) * 1.2,
                     max(sed[good_idxs][sed[good_idxs]['peak_wavelength'] < 20000]['f_lambda'] * 1.2))

    rounded_z = np.round(mosdef_obj["Z_MOSFIRE_USE"], 3)
    ax_main.text(0.02, 0.95, f'{field} {v4id}', fontsize=axisfont, transform=ax_main.transAxes)
    ax_main.text(0.02, 0.89, f'z = ' + str(rounded_z), fontsize=axisfont, transform=ax_main.transAxes)

    ax_main.set_xscale('log')
    ax_main.set_xlim(800, 45000)
    ax_zoom.set_xlim(3000, 7000)
    ax_zoom.set_xscale('log')

    # labels_x = ['']*len(ax_zoom.get_xticks(minor=True))
    # labels_x[9] = '$3 \\times 10^3$'
    # labels_x[16] = '$2 \\times 10^4$'
    # ax_zoom.set_xticklabels(labels_x, minor=True)

    labels_y = ax_zoom.get_yticks()

    ax_main.tick_params(labelsize=ticksize, size=ticks)
    ax_zoom.tick_params(labelsize=ticksize, size=ticks)

    # IMAGE PLOTTING:
    plot_image(ax_image, field, v4id, mosdef_obj)

    # SPECTRUM PLOTTING:
    if plot_spec:
        plot_spectrum(ax_main, field, v4id, mosdef_obj)
        plot_spectrum(ax_zoom, field, v4id, mosdef_obj)

    # SED FIT PLOTTING:
    if plot_fit:
        plot_sed_fit(ax_main, field, v4id)
        plot_sed_fit(ax_zoom, field, v4id)

    fig.savefig(imd.home_dir + f'/mosdef/SED_Images/{field}_{v4id}.pdf')
    plt.close('all')


def plot_sed_fit(ax, field, v4id):
    fit_func = poly_fit(field, v4id)
    fit_wavelengths = np.arange(2, 6, 0.02)
    ax.plot(10**fit_wavelengths, fit_func(
        fit_wavelengths), color='mediumseagreen')


def plot_image(ax, field, v4id, mosdef_obj):
    """Given a field, id, and axis, plot the HST image of an object

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object
    ax (plt.axis): axis to plot the image onto
    mosdef_obj (pd.Dataframe): single entry of mosdef_df from get_mosdef_obj()


    Returns:
    """
    image_loc = imd.home_dir + '/mosdef/HST_Images/' + f'{mosdef_obj["FIELD_STR"]}_f160w_{mosdef_obj["ID"]}.fits'
    hdu = fits.open(image_loc)[0]
    image_data = hdu.data
    # Center of array is 83.5,83.5
    ax.imshow(image_data[67:100, 67:100], cmap='gray', aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_spectrum(ax, field, v4id, mosdef_obj):
    """Given a field, id, and axis, plot the mosdef spectrum of an object

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object
    ax (plt.axis): axis to plot the image onto
    mosdef_obj (pd.Dataframe): single entry of mosdef_df from get_mosdef_obj()


    Returns:
    """
    # First get a list of all things in the Spectra directory
    obj_files = get_spectra_files(mosdef_obj)
    for file in obj_files:
        print(f'Plotting {file}')
        spec_loc = imd.spectra_dir + file
        hdu = fits.open(spec_loc)[1]
        spec_data = hdu.data
        wavelength = (
            1. + np.arange(hdu.header["naxis1"]) - hdu.header["crpix1"]) * hdu.header["cdelt1"] + hdu.header["crval1"]
        spec_data = clip_skylines(wavelength, spec_data)
        ax.plot(wavelength, spec_data, color='blue', lw=1)


def plot_all_seds(zobjs):
    """Given a field and id, plots the SED of a galaxy from its {field}_{v4id}_sed.csv

    Parameters:
    zobjs (list): Pass a list of tuples of the form (field, v4id)


    Returns:
    """
    counter = 0
    for obj in zobjs:
        field = obj[0]
        v4id = obj[1]
        print(f'Creating SED for {field}_{v4id}, {counter}/{len(zobjs)}')
        try:
            plot_sed(field, v4id, plot_fit=True)
        except:
            print(f'Couldnt create plot for {field}_{v4id}')
            plt.close('all')
        counter = counter + 1


def setup_spec_only(field, v4id, mosdef_obj):
    """Sets up the plot to plot ONLY the spectrum

    Parameters:


    Returns:
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_spectrum(ax, field, v4id, mosdef_obj)
    plt.show()
