# plot_mesa_hr.py
# plots the output of mesa onto an hr diagram

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
import mesa_reader as mr

# Directory with star files
stars_dir = '/Users/galaxies-air/Courses/Stars/ps4/Kurucz_Spectra/'

o_star = pd.DataFrame(fits.getdata(stars_dir+'O_Star.fits'))
a_star = pd.DataFrame(fits.getdata(stars_dir+'A_Star.fits'))
m_star = pd.DataFrame(fits.getdata(stars_dir+'M_Star.fits'))


axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


Ha_wavelength = 6560
Hb_wavelength = 4860


def plot_spectra(Ha=False, Hb=False):
    fig, ax = plt.subplots(figsize=(8, 7))

    stars = [o_star, a_star, m_star]
    star_names = ['O', 'A', 'M']
    colors = ['blue', 'black', 'red']

    for i in range(len(stars)):
        norm = stars[i]['Continuum']
        # if Ha:
        #     idx_Ha = (np.abs(stars[i]['Wavelength']-Ha_wavelength)).idxmin()
        #     norm = stars[i].iloc[idx_Ha]['Continuum']
        # if Hb:
        #     idx_Hb = (np.abs(stars[i]['Wavelength']-Hb_wavelength)).idxmin()
        #     norm = stars[i].iloc[idx_Hb]['Continuum']
        ax.plot(stars[i]['Wavelength'], stars[i]['SpecificIntensity']/norm,
                color=colors[i], label=f'{star_names[i]}_Star')

    if not Hb:
        ax.plot((Ha_wavelength, Ha_wavelength), (-100, 100),
                color='orange', ls='--', label='H$_\\alpha$')
    if not Ha:
        ax.plot((Hb_wavelength, Hb_wavelength), (-100, 100),
                color='mediumseagreen', ls='--', label='H$_\\beta$')

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('Normalized Intensity', fontsize=axisfont)
    # ax.set_yscale('log')
    ax.set_ylim(-0.05, 1.05)
    if Ha:
        ax.set_xlim(6000, 7000)
        ax.set_ylim(-0.05, 1.35)
    elif Hb:
        ax.set_xlim(4300, 5300)
        ax.set_ylim(-0.05, 1.35)
    else:
        ax.set_xscale('log')
        ax.set_xlim(1000, 30000)
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=axisfont)
    zoom_key = ''
    if Ha:
        zoom_key = '_Ha'
    if Hb:
        zoom_key = '_Hb'
    fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps4/StellarSpectra{zoom_key}.pdf')
    plt.close('all')


plot_spectra()
plot_spectra(Ha=True)
plot_spectra(Hb=True)
