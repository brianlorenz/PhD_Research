# plot_isochrone.py
# Plots the output from pypopstar_ps1.py

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt

track_dir = '/Users/galaxies-air/Courses/Stars/ps3/mist_tracks_p1/'
all_mist_tracks = os.listdir(track_dir)


axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def plot_track(ax, xdata, ydata, sun=False):
    """

    Parameters:
    ax (plt.axis): axis to plot the isochrone onto
    xdata (pd.Dataframe): column containing data to plot
    ydata (pd.Dataframe): column containing data to plot
    sun (boolean): set to true if plotting a 1M_sun star


    Returns:
    """
    if sun:
        ax.plot(xdata, ydata, color='blue', label='1 M$_\odot$')
    else:
        ax.plot(xdata, ydata, color='black')


def make_figure(key, slope):
    """

    Parameters:
    key(str): column name to plot
    slope (float): scaling relation slope found from part 1a of the problem set


    Returns:
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for file in all_mist_tracks:
        track = ascii.read(track_dir + file, header_start=11,
                           data_start=12).to_pandas()
        if file == '0010000M.track.eep':
            sun = True
        else:
            sun = False
        plot_track(ax, np.log10(track['star_age']),
                   track['log_'+key], sun=sun)
    # Plot the scaling relations
    ages = np.arange(5, 6, 0.1)
    if key == 'R':
        ages = np.arange(3.75, 4.75, 0.1)
    ax.plot(ages, ages*slope[0]+slope[1],
            color='red', label='Scaling Relation')

    # Axis settings
    ax.set_xlabel('log(Age) (Years)', fontsize=axisfont)
    ax.set_ylabel(f'log({key})', fontsize=axisfont)
    ax.legend(fontsize=axisfont)
    #ax.set_xlim(10000, 2000)
    #ax.set_ylim(0.1, 1000)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps3/p4_age_{key}.pdf')
    plt.close('all')


keys = ['L', 'R', 'Teff']
slope_dict = {
    keys[0]: (1, -3),  # (slope, yint)
    keys[1]: (-2, 9.5),
    keys[2]: (2, -6.5),
}
for key in keys:
    make_figure(key=key, slope=slope_dict[key])
