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
from query_funcs import get_zobjs
import matplotlib.pyplot as plt
import fnmatch
import initialize_mosdef_dirs as imd
import shutil
from spectra_funcs import clip_skylines, get_spectra_files, median_bin_spec, read_spectrum, smooth_spectrum


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
    if plot_spec == True:
        ax_main.set_ylim(-max(sed[good_idxs]['f_lambda'] * 1.2),
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
    redshift = mosdef_obj['Z_MOSFIRE']
    obj_files = get_spectra_files(mosdef_obj)
    print(obj_files)
    for file in obj_files:
        print(f'Plotting {file}')
        spectrum_df = read_spectrum(mosdef_obj, file)

        spectrum_df['f_lambda_clip'], spectrum_df['mask'] = clip_skylines(
            spectrum_df['rest_wavelength'], spectrum_df['f_lambda'], spectrum_df['err_f_lambda'])

        wave_bin, spec_bin = median_bin_spec(
            spectrum_df['rest_wavelength'], spectrum_df['f_lambda_clip'], binsize=100)
        smooth_spec = smooth_spectrum(
            spectrum_df['f_lambda_clip'], width=200)
        # ax.plot(wavelength / (1 + redshift),
        #         spec_data_errs, color='orange', lw=1)
        ax.plot(spectrum_df['rest_wavelength'], spectrum_df[
                'f_lambda_clip'], color='blue', lw=1)
        #ax.plot(wave_bin, spec_bin, color='orange', lw=1)
        ax.plot(spectrum_df['rest_wavelength'],
                smooth_spec, color='orange', lw=1)
        ax.scatter(spectrum_df['rest_wavelength'] * spectrum_df['mask'],
                   np.zeros(len(spectrum_df['f_lambda'])), color='red')
        # ax.set_xlim(ax_sn.get_xlim())


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
            plot_sed(field, v4id, plot_fit=True, plot_spec=True)
        except:
            print(f'Couldnt create plot for {field}_{v4id}')
            plt.close('all')
        counter = counter + 1


def setup_spec_only(zobjs):
    """Sets up the plot to plot ONLY the spectrum

    Parameters:


    Returns:
    """
    for obj in zobjs:
        field = obj[0]
        v4id = obj[1]
        mosdef_obj = get_mosdef_obj(field, v4id)

        fig, (ax, ax_sn) = plt.subplots(2, 1, figsize=(8, 9))
        plot_spectrum(ax, field, v4id, mosdef_obj, ax_sn)
        plt.show()


def sort_by_sn():
    """Sorts the galaxies by signal-to-noise ratio of their spectra

    Parameters:

    Returns:
    """

    # Make sure the directories are set up
    all_dir = imd.mosdef_dir + '/SED_Images/All_High_SN'
    some_dir = imd.mosdef_dir + '/SED_Images/Some_High_SN'
    none_dir = imd.mosdef_dir + '/SED_Images/None_High_SN'
    source_dir = imd.mosdef_dir + '/SED_Images/'
    imd.check_and_make_dir(all_dir)
    imd.check_and_make_dir(some_dir)
    imd.check_and_make_dir(none_dir)

    zobjs = get_zobjs()
    # Drop duplicates
    zobjs = list(dict.fromkeys(zobjs))
    # Remove objects with ID less than zero
    zobjs = [obj for obj in zobjs if obj[1] > 0]
    # Sort
    zobjs.sort()

    # Signal-to-noise threshold to be considered good signal to noise
    thresh = 3

    for obj in zobjs:
        mosdef_obj = get_mosdef_obj(obj[0], obj[1])
        files = get_spectra_files(mosdef_obj)

        if len(files) < 1:
            continue

        sig_noises = []
        for file in files:
            spectrum_df = read_spectrum(mosdef_obj, file)
            # Mask out skylines using 5*median of the noise
            mask_sky = spectrum_df['err_f_lambda'] < 5 * \
                np.median(spectrum_df['err_f_lambda'])
            mask_zeros = spectrum_df['f_lambda'] != 0
            mask = np.logical_and(mask_sky, mask_zeros)
            sig_noise = np.median(
                spectrum_df[mask]['f_lambda']) / np.median(spectrum_df[mask]['err_f_lambda'])
            sig_noises.append(sig_noise)
            print(f'sig_noise = {sig_noise}')

        # Now, copy the files if they have good signal to noise
        source_file = source_dir + f'{obj[0]}_{obj[1]}.pdf'
        if np.all([y > thresh for y in sig_noises]):
            # Move file
            shutil.copy(source_file, all_dir)
            print('Copied ' + f'{obj[0]}_{obj[1]}.pdf ' + 'to all_dir')
        elif np.any([y > thresh for y in sig_noises]):
            shutil.copy(source_file, some_dir)
            print('Copied ' + f'{obj[0]}_{obj[1]}.pdf ' + 'to some_dir')
        else:
            shutil.copy(source_file, none_dir)
            print('Copied ' + f'{obj[0]}_{obj[1]}.pdf ' + 'to none_dir')
