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
from plot_mist_ps3 import plot_track

track_dir = '/Users/galaxies-air/Courses/Stars/ps3/mist_tracks_p1/'
all_mist_tracks = os.listdir(track_dir)


axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def make_figure_p1():
    """

    Parameters:
    key(str): column name to plot
    slope (float): scaling relation slope found from part 1a of the problem set


    Returns:
    """
    slope_rad = 1
    slope_convect = -2/3
    yint_rad = -3
    yint_convect = 2
    fig, ax = plt.subplots(figsize=(8, 7))
    file = '0010000M.track.eep'
    track = ascii.read(track_dir + file, header_start=11,
                       data_start=12).to_pandas()
    ax.plot(10**track['log_Teff'], 10**track['log_L'],
            color='blue', label='1 M$_\odot$')
    ax.plot(10**track['log_Teff'][:200], 10**track['log_L']
            [:200], color='red', label='1 M$_\odot$, Pre-MS')
    # Plot the scaling relations
    ax.plot([100000, 3000], [0.72, 0.72], label='0.72L$_\odot$', color='black')
    # Axis settings
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 100000)
    ax.invert_xaxis()
    ax.set_xlabel('Teff (K)', fontsize=axisfont)
    ax.set_ylabel('L (L$_\odot$)', fontsize=axisfont)
    ax.legend(fontsize=axisfont, loc=1)
    #ax.set_xlim(10000, 2000)
    #ax.set_ylim(0.1, 1000)
    ax.tick_params(labelsize=ticksize, size=ticks)
    fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps3/HR_1M.pdf')
    plt.close('all')


make_figure_p1()
