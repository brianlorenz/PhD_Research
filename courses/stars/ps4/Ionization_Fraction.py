# Ionization_Fraction.py

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt


axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def ion_func(T):
    return (24.1)*(T**(3/2))*np.exp((-1.58*(10**5))/T)


def ion_saha(temp):
    return 0.5*(-ion_func(temp) +
                np.sqrt(ion_func(temp)**2+4*ion_func(temp)))


def plot_ion(use_n2=0):
    fig, ax = plt.subplots(figsize=(8, 7))

    temp = np.arange(3000, 50000, 2)

    ion_frac_saha = ion_saha(temp)

    frac_n2 = 4*np.exp((-1.18*10**5) / temp)

    ion_frac_plot = ion_frac_saha
    if use_n2:
        ion_frac_plot = frac_n2 * (1-ion_frac_saha)

    ax.plot(temp, ion_frac_plot)

    #ax.legend(fontsize=axisfont, loc=1)
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    #ax.set_xlim(200000, 1000)
    #ax.set_ylim(0.001, 10**7)
    ax.set_xlabel('Teff (K)', fontsize=axisfont)
    ax.set_ylabel('Ionization Fraction of H', fontsize=axisfont)
    if use_n2:
        key = '_n2'
    else:
        key = ''
    fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps4/ion_frac{key}.pdf')
    plt.close('all')


plot_ion()
plot_ion(use_n2=1)

print(f'Fraction of atoms in ground state of M star: {1 - ion_saha(3500)}')
