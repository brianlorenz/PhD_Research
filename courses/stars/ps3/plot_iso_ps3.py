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
from pypopstar_ps3 import evo_model_name, logAges

savenames = [f'iso_{logAge}_{evo_model_name[i]}.csv' for logAge in logAges for i in range(2)]

bar_6_6 = fits.open('iso_6.6_baraffe.fits')[1].data
bar_7_6 = fits.open('iso_7.6_baraffe.fits')[1].data
mist_6_6 = fits.open('iso_6.6_mist.fits')[1].data
mist_7_6 = fits.open('iso_7.6_mist.fits')[1].data

axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def plot_iso_cmd(ax, iso_df, color, label, hr=False):
    """

    Parameters:
    ax (plt.axis): axis to plot the isochrone onto
    iso_df (pd.DataFrame): dataframe with isochrone data
    color (str): color to plot
    label (str): label for plot
    hr (Boolean): set to True ot make an HR diagram instead of CMD


    Returns:
    """
    if hr:
        ax.plot(iso_df['Teff'],
                iso_df['L']/(3.827*10**26), color=color, label=label)
    else:
        ax.plot(iso_df['m_hst_f127m'] - iso_df['m_hst_f153m'],
                iso_df['m_hst_f153m'], color=color, label=label)


def plot_diagram(hr=False, zoom=False):
    fig, ax = plt.subplots(figsize=(8, 7))

    plot_iso_cmd(ax, bar_6_6, 'black', 'Baraffe 4Myr', hr=hr)
    plot_iso_cmd(ax, bar_7_6, 'blue', 'Baraffe 40Myr', hr=hr)
    plot_iso_cmd(ax, mist_6_6, 'red', 'Mist 4Myr', hr=hr)
    plot_iso_cmd(ax, mist_7_6, 'orange', 'Mist 40Myr', hr=hr)

    ax.legend(fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    if hr:
        ax.set_xscale('log')
        ax.set_yscale('log')
        if zoom:
            ax.set_xlim(9000, 2000)
            ax.invert_xaxis()
            ax.set_ylim(0.001, 10)
        else:
            ax.set_xlim(50000, 1000)
            ax.invert_xaxis()
            ax.set_ylim(0.001, 10**7)
        ax.set_xlabel('Teff (K)', fontsize=axisfont)
        ax.set_ylabel('L (L$_\odot$)', fontsize=axisfont)
        zoom_key = ''
        if zoom:
            zoom_key = '_zoom'
        fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps3/p4_hr{zoom_key}.pdf')
    else:
        ax.set_xlabel('F127M - F153M', fontsize=axisfont)
        ax.set_ylabel('F153M', fontsize=axisfont)
        if zoom:
            ax.set_xlim(0.1, 0.6)
            ax.set_ylim(10, 20)
        ax.invert_yaxis()
        zoom_key = ''
        if zoom:
            zoom_key = '_zoom'
        fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps3/p4_cmd{zoom_key}.pdf')
    plt.close('all')


plot_diagram()
plot_diagram(hr=True)
plot_diagram(hr=True, zoom=True)
plot_diagram(zoom=True)
