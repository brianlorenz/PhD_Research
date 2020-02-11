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
import matplotlib.pyplot as plt


def populate_main_axis(ax, sed, good_idxs, axisfont, ticksize, ticks):
    ax.errorbar(sed[good_idxs]['peak_wavelength'], sed[good_idxs]['f_lambda'], yerr=[
                sed[good_idxs]['err_f_lambda'], sed[good_idxs]['err_f_lambda']], ls='None', color='black', marker='o')
    outliers = sed[np.logical_not(good_idxs)]
    [ax.axvspan(outliers.iloc[i]['peak_wavelength']*0.99,
                outliers.iloc[i]['peak_wavelength']*1.01, facecolor='r', alpha=0.5) for i in range(len(outliers))]
    ax.set_xlabel('log($\lambda$) ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('f$_{\lambda}$', fontsize=axisfont)


def plot_sed(field, v4id):
    """Given a field and id, read in the sed and create a plot of it

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object


    Returns:
    """
    sed_location = f'/Users/galaxies-air/mosdef/sed_csvs/{field}_{v4id}_sed.csv'
    sed = ascii.read(sed_location).to_pandas()

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
    # Bottom limit is either 0 or 1.2* most negative value. Upper limit is 1.2* most positive value

    ax_main.set_ylim(min(0, min(sed[good_idxs]['f_lambda']))*1.2,
                     max(sed[good_idxs]['f_lambda']*1.2))
    ax_zoom.set_ylim(min(0, min(sed[good_idxs][sed[good_idxs]['peak_wavelength'] < 20000]['f_lambda']))*1.2,
                     max(sed[good_idxs][sed[good_idxs]['peak_wavelength'] < 20000]['f_lambda']*1.2))

    rounded_z = np.round(mosdef_obj.iloc[0]["Z_MOSFIRE_USE"], 3)
    ax_main.text(0.02, 0.95, f'{field} {v4id}', fontsize=axisfont, transform=ax_main.transAxes)
    ax_main.text(0.02, 0.89, f'z = ' + str(rounded_z), fontsize=axisfont, transform=ax_main.transAxes)

    ax_main.set_xscale('log')
    ax_main.set_xlim(3*10**3, 10**5)
    ax_zoom.set_xlim(3*10**3, 2.5*10**4)
    ax_zoom.set_xscale('log')

    labels_x = ['']*len(ax_zoom.get_xticks(minor=True))
    labels_x[9] = '$3 \\times 10^3$'
    labels_x[16] = '$2 \\times 10^4$'
    ax_zoom.set_xticklabels(labels_x, minor=True)

    labels_y = ax_zoom.get_yticks()

    # plt.xticks([4*10**3, 10**4, 2*10**4],
    #           ['$4\\times 10^3$', '$10^4$', '$2\\times 10^4$'])
    # ax_zoom.set_xticks([3000, 10000, 20000])
    # ax_zoom.set_xticklabels(
    #    ['', '$4\\times 10^3$', '', '$10^4$', '$2\\times 10^4$'])

    ax_main.tick_params(labelsize=ticksize, size=ticks)
    ax_zoom.tick_params(labelsize=ticksize, size=ticks)

    # IMAGE PLOTTING:
    plot_image(ax_image, field, v4id, mosdef_obj)

    fig.savefig('/Users/galaxies-air/Desktop/test.pdf')
    plt.close('all')


def plot_image(ax, field, v4id, mosdef_obj):
    """Given a field, id, and axis, plot the HST image of an object

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object
    ax (plt.axis): axis to plot the image onto
    mosdef_obj (pd.Dataframe): single entry of mosdef_df from get_mosdef_obj()


    Returns:
    """
    image_loc = '/Users/galaxies-air/mosdef/HST_Images/'+f'{mosdef_obj.iloc[0]["FIELD_STR"]}_f160w_{mosdef_obj.iloc[0]["ID"]}.fits'
    hdu = fits.open(image_loc)[0]
    image_data = hdu.data
    # Center of array is 83.5,83.5
    ax.imshow(image_data[70:97, 70:97], cmap='gray', aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# STILL NEED TO TRANSFER THE SPECTRA FILES
def plot_spectrum(ax, field, v4id, mosdef_obj):
    """Given a field, id, and axis, plot the mosdef spectrum of an object

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object
    ax (plt.axis): axis to plot the image onto
    mosdef_obj (pd.Dataframe): single entry of mosdef_df from get_mosdef_obj()


    Returns:
    """
    spec_loc = '/Users/galaxies-air/mosdef/Spectra/'+f'{mosdef_obj.iloc[0]["FIELD_STR"]}_f160w_{mosdef_obj.iloc[0]["ID"]}.fits'
    hdu = fits.open(image_loc)[0]
    spec_data = hdu.data
    # Center of array is 83.5,83.5
    ax.plot()


def get_mosdef_obj(field, v4id):
    """Given a field and id, find the object in the mosdef_df dataframe

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object

    Returns:
    mosdef_obj (pd.DataFrame): Datatframe with one entry corresponding to the current object
    """
    mosdef_obj = mosdef_df[np.logical_and(
        mosdef_df['FIELD_STR'] == field, mosdef_df['V4ID'] == v4id)]
    # There should be a unique match - exit with an error if not
    if len(mosdef_obj) < 1:
        sys.exit('No match found on FIELD_STR and V4ID')
    # If there's a duplicate, take the first one
    if len(mosdef_obj) > 1:
        mosdef_obj = mosdef_obj.iloc[0]
        # WHAT TO DO HERE WHEN YOU FIND A REPEAT? NEED TO STORE AND MOVE ONE
        print('Duplicate obj, taking the first instance')
    return mosdef_obj
