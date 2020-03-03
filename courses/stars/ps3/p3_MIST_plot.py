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


def make_figure_p3():
    """

    Parameters:
    key(str): column name to plot
    slope (float): scaling relation slope found from part 1a of the problem set


    Returns:
    """
    slope_rad = 1
    slope_convect = -2/3
    yint_rad = -7.75
    yint_convect = 5.15
    fig, ax = plt.subplots(figsize=(8, 7))
    file = '0010000M.track.eep'
    track = ascii.read(track_dir + file, header_start=11,
                       data_start=12).to_pandas()
    idx = np.logical_and(track['star_age'] > 10**7.43,
                         track['star_age'] < 10**7.6)
    ax.plot(np.log10(track['star_age']), track['log_L'],
            color='blue', label='1 M$_\odot$')
    ax.plot(np.log10(track['star_age'][idx]), track['log_L'][idx],
            color='orange', label='Henyey Hook')
    # Plot the scaling relations
    ages = np.arange(7.4, 7.6, 0.01)
    ax.plot(ages, ages*slope_rad+yint_rad,
            color='red', label='Fully Radiative Slope')
    ax.plot(ages, ages*slope_convect+yint_convect,
            color='black', label='Fully Convective Slope')
    # Axis settings
    ax.set_xlabel('log(Age) (Years)', fontsize=axisfont)
    ax.set_ylabel(f'log(L)', fontsize=axisfont)
    ax.legend(fontsize=axisfont)
    #ax.set_xlim(10000, 2000)
    #ax.set_ylim(0.1, 1000)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps3/p3.pdf')
    plt.close('all')


make_figure_p3()
