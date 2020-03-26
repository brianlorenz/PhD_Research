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

zsun = fits.open('solar.fits')[1].data
zlow = fits.open('lowz.fits')[1].data

axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def plot_diagram(iso_df1=zsun, iso_df2=zlow, zoom=0):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(iso_df1['Teff'],
            iso_df1['L']/(3.827*10**26), color='blue', label='Solar Metallicity')
    ax.plot(iso_df2['Teff'],
            iso_df2['L']/(3.827*10**26), color='orange', label='Z=0.1Z$_\odot$')

    ax.legend(fontsize=axisfont, loc=1)
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if zoom:
        ax.set_xlim(9000, 2000)
        ax.invert_xaxis()
        ax.set_ylim(0.001, 40000)
    else:
        ax.set_xlim(200000, 1000)
        ax.invert_xaxis()
        ax.set_ylim(0.001, 10**7)
    ax.set_xlabel('Teff (K)', fontsize=axisfont)
    ax.set_ylabel('L (L$_\odot$)', fontsize=axisfont)
    zoom_key = ''
    if zoom:
        zoom_key = '_zoom'
    fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps4/iso_hr{zoom_key}.pdf')
    plt.close('all')


plot_diagram()
plot_diagram(zoom=1)
